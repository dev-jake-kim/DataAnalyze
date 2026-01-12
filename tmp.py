import matplotlib.pyplot as plt

# 1. 데이터 입력
data = [
    (0.8, 0.5265),(0.5, 0.4607), (0.2, 0.4168), (0.1, 0.4017), (0, 0.399999)
]

# 2. x축 기준으로 데이터 정렬 (꺾은선이 꼬이지 않게 하기 위함)
data.sort(key=lambda x: x[0])

# 3. x, y 데이터 분리
x_values = [d[0] for d in data]
y_values = [d[1] for d in data]

# 4. 그래프 생성
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', linewidth=2, markersize=6)

# 그래프 정보 설정
plt.title('Calibration weight Line Plot', fontsize=14)
plt.xlabel('alpha', fontsize=12)
plt.ylabel('covered loss', fontsize=12)

# y값이 급격히 줄어들므로 로그 스케일을 적용하면 작은 값의 변화도 잘 보입니다.
# plt.yscale('log') # 필요 시 주석을 해제하여 사용하세요.

plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(x_values) # 모든 x값이 표시되도록 설정

plt.show()