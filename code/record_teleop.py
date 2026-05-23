"""
텔레오퍼레이션 데이터 녹화 스크립트
- 모든 에피소드를 data_record 폴더 하나에 누적 저장합니다.
- 실행할 때마다 이전 에피소드에 이어서 추가 저장됩니다.

실행 방법:
    python record_teleop.py

조작 방법:
    - 녹화 완료 후 저장: 오른쪽 화살표 키 (→)
    - 현재 에피소드 취소 후 재녹화: 왼쪽 화살표 키 (←)  ← 주의: 삭제 후 재녹화
    - 전체 종료 (저장됨): ESC 또는 Ctrl+C
"""

import os
import sys
import subprocess

# ─────────────────────────────────────────────────────────────
# ★ 설정 구역 (필요에 따라 수정하세요) ★
# ─────────────────────────────────────────────────────────────
FOLLOWER_PORT   = "COM4"        # 팔로워 로봇 포트
LEADER_PORT     = "COM3"        # 리더(텔레오프) 포트
CAM_FOLLOWER_IDX = 0            # 손목 카메라 인덱스
CAM_TOP_IDX     = 1            # 탑뷰 카메라 인덱스

SAVE_DIR        = os.path.expanduser("~/data_record") # 저장 폴더 이름 (한글 경로 깨짐 방지를 위해 사용자 홈 폴더에 저장)
REPO_ID         = "myuser/pick_teleop"  # 데이터셋 식별자
TASK_NAME       = "pick"        # 작업 이름

FPS             = 30            # 녹화 FPS
EPISODE_TIME_S  = 1800          # 에피소드 최대 녹화 시간 (초) — 기본 30분
RESET_TIME_S    = 30            # 에피소드 사이 리셋 시간 (초)
NUM_EPISODES    = 999           # 한 세션에 최대 녹화할 에피소드 수
# ─────────────────────────────────────────────────────────────


def main():
    print("=" * 55)
    print("🤖  텔레오퍼레이션 데이터 녹화 프로그램  🤖")
    print("=" * 55)

    resume_mode = False

    # data_record 폴더 처리
    if os.path.exists(SAVE_DIR):
        # 폴더 내 파일 존재 여부 확인
        files = os.listdir(SAVE_DIR)
        if len(files) == 0:
            # 완전히 빈 폴더인 경우: LeRobotDataset.create 가 충돌하지 않도록 삭제
            os.rmdir(SAVE_DIR)
            print(f"🧹 빈 저장 폴더 발견 및 자동 삭제 완료 (새로 생성 예정)")
        else:
            # 이미 기존 데이터가 있는 경우: 이어서 기록하도록 resume 모드 활성화
            resume_mode = True
            print(f"📂 기존 데이터가 들어있는 폴더 감지: {SAVE_DIR}/ (이어서 녹화합니다.)")
    else:
        print(f"📁 새 저장 폴더가 생성됩니다: {SAVE_DIR}/")

    print()
    print("💡 준비 사항:")
    print("   1. 팔로워 로봇과 리더 암을 시작 위치로 정렬해 주세요.")
    print("   2. 카메라 화각과 조명을 확인해 주세요.")
    print()
    print("🎮 조작 방법:")
    print("   → (오른쪽 화살표) : 현재 에피소드 저장 후 다음 에피소드로")
    print("   ← (왼쪽 화살표)  : 현재 에피소드 취소 후 재녹화")
    print("   ESC / Ctrl+C     : 지금까지 저장된 에피소드 유지 후 종료")
    print("=" * 55)

    try:
        input("👉 준비가 완료되었다면 [Enter] 키를 누르세요. 녹화가 시작됩니다...")
    except KeyboardInterrupt:
        print("\n👋 프로그램을 종료합니다.")
        return

    # lerobot_record.py 호출 명령어 구성
    cmd = [
        sys.executable,
        "lerobot/src/lerobot/scripts/lerobot_record.py",

        # 로봇 설정
        "--robot.type=so100_follower",
        f"--robot.port={FOLLOWER_PORT}",
        "--robot.calibration_dir=calibration_backup/robots/so_follower",

        # 카메라 설정 (두 대)
        f"--robot.cameras={{cam_follower: {{type: opencv, index_or_path: {CAM_FOLLOWER_IDX}, "
        f"width: 640, height: 480, fps: {FPS}, backend: 700}}, "
        f"cam_top: {{type: opencv, index_or_path: {CAM_TOP_IDX}, "
        f"width: 640, height: 480, fps: {FPS}, backend: 700}}}}",

        # 텔레오프 설정
        "--teleop.type=so100_leader",
        f"--teleop.port={LEADER_PORT}",
        "--teleop.calibration_dir=calibration_backup/teleoperators/so_leader",

        # 데이터셋 설정
        f"--dataset.root={SAVE_DIR}",
        f"--dataset.repo_id={REPO_ID}",
        f"--dataset.single_task={TASK_NAME}",
        f"--dataset.num_episodes={NUM_EPISODES}",
        "--dataset.push_to_hub=false",
        f"--dataset.episode_time_s={EPISODE_TIME_S}",
        f"--dataset.reset_time_s={RESET_TIME_S}",
        f"--dataset.fps={FPS}",
    ]

    if resume_mode:
        cmd.append("--resume=True")

    print(f"\n🚀 녹화 시작! 데이터는 '{SAVE_DIR}/' 폴더에 저장됩니다.")
    print("-" * 55)

    try:
        subprocess.run(cmd, check=True)
        print("\n" + "=" * 55)
        print(f"🎉 녹화 완료! 저장 위치: {SAVE_DIR}/")
        print("=" * 55)
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 55)
        if e.returncode == 0:
            print("✅ 정상 종료되었습니다.")
        else:
            print(f"⚠️  경고: 녹화 중 오류 또는 강제 종료가 발생했습니다. (코드: {e.returncode})")
            print(f"   저장된 에피소드는 {SAVE_DIR}/ 에 남아 있습니다.")
        print("=" * 55)
    except KeyboardInterrupt:
        print("\n👋 녹화가 종료되었습니다.")
        print(f"   저장된 데이터는 '{SAVE_DIR}/' 폴더를 확인하세요.")


if __name__ == "__main__":
    main()
