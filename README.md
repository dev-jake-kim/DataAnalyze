# DataAnalyze

`create_graph.py`는 원본 택시 호출 데이터를 읽어 전처리하고, 시간-공간 수요 grid를 만든 뒤, 수요가 높은 패치를 노드로 선택하여 최종 `graph_data.json`을 생성한다. 이 문서는 원본 데이터 스키마, 결측치 점검, 필요한 속성 추출, 전처리, 시각화, 그래프 생성까지의 전체 과정을 정리한 것이다.

## 1. 입력 데이터

주요 입력 파일은 아래 3개다.

- `data/origin_data.csv`: 원본 호출 데이터
- `data/UPIS_C_UQ111.shp`: 토지이용 polygon shapefile
- `data/poi_data.csv`: POI 데이터

`origin_data.csv`는 `cp949` 인코딩으로 읽으며, 좌표는 소수점이 생략된 WGS84 정수 형식이다.

- 예시: `128950739 -> 128.950739`
- `xpos`, `ypos`, `dest_xpos`, `dest_ypos` 모두 같은 규칙을 따른다.

## 2. 원본 데이터 스키마

현재 `data/origin_data.csv` 기준:

- 총 행 수: `727,026`
- 총 컬럼 수: `40`
- 인코딩: `cp949`

### 2.1 전체 컬럼 목록

```text
allocid
clientid
drvseq
call_date
call_address1
call_address2
call_address3
xpos
ypos
dest_xpos
dest_ypos
call_state
call_process
send_level
opname
send_time
recv_time
repeat_time
card_type
cancel_type
return_flag
drv_distance
drv_cash
drv_addedcash
payments
req_call_state
board_time
leave_time
agent_alloc_cnt
safe
driving_alloc
citycode0
citycode1
citycode2
citycode3
customer_call_fail
vehproblem
local_idx
ClientType
av_carno
```

### 2.2 그래프 생성에서 실제로 사용하는 핵심 속성

| 컬럼 | 예시 dtype | 역할 |
| --- | --- | --- |
| `clientid` | `int64` | 중복 제거 기준 키 |
| `call_date` | `object` -> `datetime64[ns]` | 시간 파싱 후 시계열 grid 생성에 사용 |
| `xpos` | `int64` | 출발지 경도 좌표 |
| `ypos` | `int64` | 출발지 위도 좌표 |
| `dest_xpos` | `int64` | 도착지 경도 좌표, OD 추출 시 사용 |
| `dest_ypos` | `int64` | 도착지 위도 좌표, OD 추출 시 사용 |

나머지 컬럼들은 원본에는 존재하지만 현재 `create_graph.py` 기준 그래프 생성 핵심 로직에서는 직접 사용하지 않는다.

## 3. 결측치 확인 결과

원본 CSV 전체(`727,026`행)를 기준으로 결측치를 집계한 결과는 아래와 같다.

### 3.1 결측치가 있는 컬럼

| 컬럼 | 결측 개수 | 결측 비율 |
| --- | ---: | ---: |
| `call_address3` | `685,611` | `94.30%` |
| `av_carno` | `185,225` | `25.48%` |
| `call_address2` | `67` | `0.01%` |

### 3.2 핵심 컬럼 결측치 확인

그래프 생성에 필요한 핵심 컬럼은 결측치가 없다.

| 컬럼 | 결측 개수 |
| --- | ---: |
| `xpos` | `0` |
| `ypos` | `0` |
| `clientid` | `0` |
| `call_date` | `0` |
| `dest_xpos` | `0` |
| `dest_ypos` | `0` |

또한 `call_date`를 `'%Y-%m-%d %H:%M:%S.%f'` 형식으로 파싱했을 때 변환 실패(`NaT`)는 `0`건이다.

## 4. 결측치 처리 원칙

이 프로젝트는 결측치가 있다고 해서 모든 행을 일괄 삭제하지 않는다. 대신 "실제로 사용하는 속성에 결측이 있는지"를 먼저 확인하고, 필요한 단계에서만 제외한다.

- 전처리 시작 시 `clientid`, `call_date`, `xpos`, `ypos`의 존재 여부를 확인한다.
- 이 4개 핵심 컬럼은 현재 데이터셋에서 결측이 없으므로, 전처리 초기에 `dropna()`를 수행하지 않는다.
- `dest_xpos`, `dest_ypos`는 OD 흐름 추출에만 필요하므로, OD 계산 단계에서만 `dropna(subset=['dest_xpos', 'dest_ypos'])`를 적용한다.
- 사용하지 않는 주소/차량번호 계열 결측치는 그래프 생성 품질에 직접 영향이 없으므로 현재 파이프라인에서는 제거 대상이 아니다.

## 5. 속성 추출과 전처리 전체 흐름

아래 순서는 `create_graph.py` 기준이다.

### Step 1. 원본 CSV 로드 및 핵심 컬럼 확인

- 파일을 `cp949`로 읽는다.
- 필수 컬럼 `['xpos', 'ypos', 'clientid', 'call_date']`가 존재하는지 검사한다.

### Step 2. 시간 컬럼 파싱

- `call_date`를 `datetime`으로 변환한다.
- 형식: `'%Y-%m-%d %H:%M:%S.%f'`

### Step 3. 공간 범위 필터링

- `remove_other_region()`으로 울산 범위 밖 데이터를 제거한다.
- 사용 범위:
  - `US_RANGE_X = (128_950739, 129_500739)`
  - `US_RANGE_Y = (35_756099, 35_305729)`

### Step 4. 중복 호출 제거

- `eleminate_duplicates()`를 사용한다.
- 같은 `clientid`가 `10분(600초)` 이내에 다시 등장하면 중복으로 보고 제거한다.

### Step 5. Crop 기반 outlier 제거

- `crop_filter(threshold_rate=0.85, step_rate=0.001)`를 적용한다.
- 중심에서 멀리 떨어진 포인트를 반복적으로 제거해 전체 데이터의 `85%` 수준까지 축소한다.

### Step 6. 필요한 속성만 추출

전처리 후 실제 공간 위치 생성에는 아래 속성만 사용한다.

- `xpos`
- `ypos`

시간 정보가 필요한 후속 처리에는 아래 속성도 함께 유지한다.

- `clientid`
- `call_date`

OD 흐름 계산에는 아래 속성도 선택적으로 사용한다.

- `dest_xpos`
- `dest_ypos`

### Step 7. GeoDataFrame 생성

- `xpos`, `ypos`를 `1,000,000`으로 나누어 WGS84 좌표(`lon`, `lat`)를 복원한다.
- `EPSG:4326`에서 `EPSG:5174`로 변환한다.
- 이후 공간 grid, patch, land-use 매핑은 모두 `EPSG:5174` 기준으로 수행한다.

### Step 8. Step 1 결과 시각화

- crop이 끝난 수요 포인트를 `GeoPandas`로 시각화한다.
- 파일: `output/step1_cropped_demand_points.png`
- `contextily`가 설치되어 있으면 basemap도 함께 표시한다.

### Step 9. 시계열 demand grid 생성

- `GRID_SIZE = 100`m 기준으로 `temporal_grid(T, H, W)`를 만든다.
- 각 레코드는 `(hour, row, col)`로 매핑된다.
- 시간축은 데이터가 존재하는 시점만 쓰지 않고, `start_hour ~ end_hour` 전체를 `pd.date_range(..., freq='h')`로 연속 생성한다.
- 따라서 데이터가 없는 시간도 `0 demand`로 포함된다.

### Step 10. 총 수요 grid 계산

- `sum_grid = temporal_grid.sum(axis=0)`
- 시계열을 공간별 총 수요로 압축한다.

### Step 11. 수요가 높은 패치 선택

- `PATCH_SIZE = 7`
- `N_PATCHES = 50`
- `PADDING_SIZE = 3`

`get_patch()`는 다음을 수행한다.

- `sum_grid`에서 `7x7` 패치를 선택한다.
- 패치끼리 겹치지 않도록 제약을 둔다.
- 총 수요 합이 최대가 되도록 최적화한다.
- 각 패치의 내부 셀(`cells`)과 주변 셀(`near_cells`)을 함께 만든다.

주변 셀 개수는 경계 clipping이 없다면 다음과 같다.

- `(PATCH_SIZE + 2 * PADDING_SIZE)^2 - PATCH_SIZE^2`
- 현재 설정에서는 `(7 + 6)^2 - 7^2 = 120`

### Step 12. 토지이용 grid 생성

- `UPIS_C_UQ111.shp`를 읽고 `ATRB_SE`를 grid 셀 중심점 기준으로 매핑한다.
- `UQA1xx ~ UQA5xx`는 각각 `1 ~ 5`로 압축한다.

### Step 13. POI 집계 및 노드 구성 생성

- `poi_data.csv`에서 `좌표정보x(epsg5174)`, `좌표정보y(epsg5174)`, `개방서비스아이디`를 사용한다.
- 각 노드에 대해 아래를 계산한다.
  - 평균 위경도
  - 토지이용 구성(`composition.land_use`)
  - POI 구성(`composition.poi`)
  - 내부 셀 목록(`cells`)
  - 주변 셀 목록(`near_cells`)

### Step 14. 노드 수요, 주변 수요, OD, 시간 특성 추출

- `extract_node_demands()`로 노드 내부 수요를 추출한다.
- 같은 함수를 `near_cells`에도 적용해 `near_demands`를 만든다.
- `dest_xpos`, `dest_ypos`가 있으면 OD 흐름을 계산한다.
- 시간 특성(`day`, `time`, `holiday`)을 생성한다.

### Step 15. JSON 저장

최종 결과는 `output/graph_data.json`에 저장된다.

## 6. 최종 산출물 구조

`graph_data.json`의 최상위 구조는 아래와 같다.

```json
{
  "nodes": [
    {
      "node_id": 0,
      "lat": 35.0,
      "lon": 129.0,
      "composition": {
        "land_use": {},
        "poi": {}
      },
      "cells": [],
      "near_cells": [],
      "size": 0.49
    }
  ],
  "x": [
    {
      "demand": [],
      "near_demands": [],
      "day": "Mon",
      "time": 0,
      "holiday": false,
      "OD": []
    }
  ]
}
```

의미는 다음과 같다.

- `nodes`: 선택된 패치 기반 노드 메타데이터
- `x[t].demand`: 시각 `t`에서 각 노드 내부 수요
- `x[t].near_demands`: 시각 `t`에서 각 노드 주변 수요
- `x[t].day`, `x[t].time`, `x[t].holiday`: 시간 특성
- `x[t].OD`: 시각 `t`에서 노드 간 OD 흐름

## 7. 실행 방법

가상환경 기준:

```bash
.venv/bin/python create_graph.py
```

또는 시스템 Python 사용 시:

```bash
python3 create_graph.py
```

## 8. 주요 출력 파일

- `output/graph_data.json`: 최종 그래프 데이터
- `output/step1_cropped_demand_points.png`: Step 1 crop 이후 수요 포인트
- `output/patch_near_demands.png`: 선택된 패치와 주변 셀 시각화
- `output/node_demand_density_curves.png`: 노드 평균 수요 / sigma density curve
- `output/landuse_grid.npy`: 토지이용 grid

## 9. 요약

이 프로젝트의 핵심은 아래 3가지다.

- 원본 CSV에서 실제로 필요한 컬럼만 골라 전처리한다.
- 핵심 컬럼의 결측과 시간 파싱 실패를 먼저 확인하고, 필요한 단계에서만 제거한다.
- 전처리된 호출 데이터를 시간-공간 graph 형태(`nodes`, `demand`, `near_demands`, `OD`)로 변환한다.
