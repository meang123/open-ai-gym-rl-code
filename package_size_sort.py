# import os
# import subprocess
#
# # 현재 활성화된 conda 환경의 base 디렉토리 찾기
# conda_base = subprocess.check_output("conda info --base", shell=True).decode().splitlines()[0].strip()
#
# # site-packages 경로 생성
# site_packages_path = os.path.join(conda_base, 'Lib', 'site-packages')
#
# # 패키지 크기 계산
# package_sizes = {}
# for pkg in os.listdir(site_packages_path):
#     pkg_path = os.path.join(site_packages_path, pkg)
#     if os.path.isdir(pkg_path):
#         total_size = 0
#         for dirpath, dirnames, filenames in os.walk(pkg_path):
#             for f in filenames:
#                 fp = os.path.join(dirpath, f)
#                 total_size += os.path.getsize(fp)
#         package_sizes[pkg] = total_size
#
# # 용량별 정렬
# sorted_packages = sorted(package_sizes.items(), key=lambda x: x[1], reverse=True)
#
# # 결과 출력
# for pkg, size in sorted_packages:
#     print(f"{pkg}: {size / (1024 ** 2):.2f} MB")


import os
import subprocess

# conda 환경 리스트에서 특정 환경의 경로 찾기
conda_envs = subprocess.check_output("conda env list", shell=True).decode().splitlines()
env_name = "RL"
env_path = None

for line in conda_envs:
    if env_name in line:
        env_path = line.split()[-1]  # 경로는 마지막에 위치

if env_path:
    site_packages_path = os.path.join(env_path, 'Lib', 'site-packages')

    # 패키지 크기 계산
    package_sizes = {}
    for pkg in os.listdir(site_packages_path):
        pkg_path = os.path.join(site_packages_path, pkg)
        if os.path.isdir(pkg_path):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(pkg_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            package_sizes[pkg] = total_size

    # 용량별 정렬
    sorted_packages = sorted(package_sizes.items(), key=lambda x: x[1], reverse=True)

    # 결과 출력
    for pkg, size in sorted_packages:
        print(f"{pkg}: {size / (1024 ** 2):.2f} MB")
else:
    print(f"Environment '{env_name}' not found.")
