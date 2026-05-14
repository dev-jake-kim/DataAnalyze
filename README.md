# DataAnalyze

이 프로젝트는 택시 호출 원본 데이터를 전처리해서 시공간 수요 그래프(`graph_data.json`)로 변환한다. 현재 지원하는 실행 진입점은 `create_graph.py` 하나이며, 실제 구현은 `graph_pipeline/` 패키지로 분리되어 있다.

## Entry Point

```bash
python3 create_graph.py
```

가상환경을 쓰는 경우:

```bash
.venv/bin/python create_graph.py
```

## Module Layout

- [create_graph.py](/Users/infolab/VscodeProjects/DataAnalyze/create_graph.py): 얇은 실행 엔트리포인트
- [graph_pipeline/config.py](/Users/infolab/VscodeProjects/DataAnalyze/graph_pipeline/config.py): 런타임 설정값
- [graph_pipeline/filters.py](/Users/infolab/VscodeProjects/DataAnalyze/graph_pipeline/filters.py): 지역 필터, 중복 제거, crop 필터
- [graph_pipeline/preprocess.py](/Users/infolab/VscodeProjects/DataAnalyze/graph_pipeline/preprocess.py): CSV 로드, 스키마 확인, 결측치 점검, 시간 파싱, GeoDataFrame 변환
- [graph_pipeline/build.py](/Users/infolab/VscodeProjects/DataAnalyze/graph_pipeline/build.py): temporal grid, patch 선택, land use/POI 집계, node/OD 생성
- [graph_pipeline/export.py](/Users/infolab/VscodeProjects/DataAnalyze/graph_pipeline/export.py): 최종 JSON 저장
- [graph_pipeline/visualization.py](/Users/infolab/VscodeProjects/DataAnalyze/graph_pipeline/visualization.py): basemap-aware 시각화 helper

## Inputs

- `data/origin_data.csv`: 원본 호출 데이터
- `data/UPIS_C_UQ111.shp`: 토지이용 polygon
- `data/poi_data.csv`: POI 데이터

`origin_data.csv`는 `cp949`로 읽고, 좌표는 소수점이 생략된 WGS84 정수 형식이다.

- 예: `128950739 -> 128.950739`
- `xpos`, `ypos`, `dest_xpos`, `dest_ypos` 모두 같은 규칙을 따른다.

## Raw Schema Summary

현재 `data/origin_data.csv` 기준:

- 총 행 수: `727,026`
- 총 컬럼 수: `40`

전체 컬럼:

```text
allocid, clientid, drvseq, call_date, call_address1, call_address2,
call_address3, xpos, ypos, dest_xpos, dest_ypos, call_state, call_process,
send_level, opname, send_time, recv_time, repeat_time, card_type,
cancel_type, return_flag, drv_distance, drv_cash, drv_addedcash, payments,
req_call_state, board_time, leave_time, agent_alloc_cnt, safe,
driving_alloc, citycode0, citycode1, citycode2, citycode3,
customer_call_fail, vehproblem, local_idx, ClientType, av_carno
```

그래프 생성에 직접 사용하는 핵심 컬럼:

- `clientid`
- `call_date`
- `xpos`
- `ypos`
- `dest_xpos`
- `dest_ypos`

## Missing-Value Check

원본 CSV 전체 기준 결측치가 있는 컬럼:

| column | missing | ratio |
| --- | ---: | ---: |
| `call_address3` | `685,611` | `94.30%` |
| `av_carno` | `185,225` | `25.48%` |
| `call_address2` | `67` | `0.01%` |

핵심 컬럼 결측치:

| column | missing |
| --- | ---: |
| `xpos` | `0` |
| `ypos` | `0` |
| `clientid` | `0` |
| `call_date` | `0` |
| `dest_xpos` | `0` |
| `dest_ypos` | `0` |

`call_date`를 `'%Y-%m-%d %H:%M:%S.%f'`로 파싱했을 때 `NaT`는 `0`건이다.

## Preprocessing Rules

결측치가 있다고 해서 모든 행을 일괄 삭제하지는 않는다. 필요한 속성에 대해서만 확인하고, 필요한 단계에서만 제외한다.

- 시작 시 필수 컬럼 `xpos`, `ypos`, `clientid`, `call_date`의 존재를 검사한다.
- `call_date`는 `datetime`으로 파싱한다.
- `remove_other_region()`으로 울산 범위 밖 데이터를 제거한다.
- `eleminate_duplicates()`로 같은 `clientid`의 10분 이내 재등장을 제거한다.
- `crop_filter(threshold_rate=0.85, step_rate=0.001)`로 outlier를 제거한다.
- `dest_xpos`, `dest_ypos`는 OD 계산 단계에서만 사용하므로, 그 시점에서만 `dropna(subset=[...])`를 적용한다.

## End-to-End Pipeline

1. 원본 CSV 로드 및 필수 컬럼 확인
2. `call_date` 파싱
3. 공간 범위 필터링
4. 중복 호출 제거
5. crop 기반 outlier 제거
6. `xpos`, `ypos`를 WGS84로 복원한 뒤 `EPSG:5174`로 투영
7. Step 1 결과 포인트 시각화
8. `temporal_grid(T, H, W)` 생성
9. `sum_grid` 계산
10. `7x7` 패치 `50`개 선택
11. `padding_size=3` 기준 `near_cells` 생성
12. shapefile을 grid에 매핑해 `landuse_grid` 생성
13. POI 집계 및 node composition 계산
14. node demand / near demand / OD / temporal feature 생성
15. `graph_data.json` 저장

## Key Runtime Settings

기본 설정은 [graph_pipeline/config.py](/Users/infolab/VscodeProjects/DataAnalyze/graph_pipeline/config.py)에 모여 있다.

- `grid_size = 100`
- `target_crs = EPSG:5174`
- `n_patches = 50`
- `patch_size = 7`
- `padding_size = 3`

경계 clipping이 없다면 노드당 `near_cells` 수는:

```text
(patch_size + 2 * padding_size)^2 - patch_size^2 = 120
```

## Outputs

- `output/graph_data.json`: 최종 그래프 데이터
- `output/step1_cropped_demand_points.png`: Step 1 이후 포인트 지도
- `output/patch_near_demands.png`: 패치/주변 셀 시각화
- `output/node_demand_density_curves.png`: node mean demand / sigma density curve
- `output/landuse_grid.npy`: 토지이용 grid

## Output JSON Shape

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

의미:

- `nodes`: 선택된 패치 기반 node 메타데이터
- `x[t].demand`: 시각 `t`에서 각 node 내부 수요
- `x[t].near_demands`: 시각 `t`에서 각 node 주변 수요
- `x[t].day`, `x[t].time`, `x[t].holiday`: 시간 특성
- `x[t].OD`: 시각 `t`의 node 간 OD 흐름
