import os
import sys
import re
import time
import numpy as np
import torch
import torch.nn as nn

from huggingface_hub import InferenceClient

try:
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower.config_so_follower import SO100FollowerConfig
    from lerobot.utils.robot_utils import precise_sleep
    LEROBOT_AVAILABLE = True
except Exception:
    LEROBOT_AVAILABLE = False


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# HF_TOKEN = "PUT_YOUR_HUGGINGFACE_TOKEN_HERE"
HF_TOKEN = os.getenv("HF_TOKEN") ## Hugging Face Token 넣으시면 됩니다.
MODEL_ID = "Qwen/Qwen3-8B"

client = InferenceClient(api_key=HF_TOKEN) if HF_TOKEN else None


SUBGOAL_SCHEMA = [
    "MOVE_TO_SOURCE_OBJECT",
    "PICK_SOURCE_OBJECT",
    "MOVE_SOURCE_TO_TARGET",
    "POUR_SOURCE_INTO_TARGET",
    "MOVE_SOURCE_BACK",
    "RELEASE_SOURCE_OBJECT",
    "MOVE_TO_TOOL_OBJECT",
    "PICK_TOOL_OBJECT",
    "MOVE_TOOL_TO_TARGET",
    "STIR_TARGET",
    "MOVE_TOOL_BACK",
    "RELEASE_TOOL_OBJECT",
]

SUBGOAL_TO_INDEX = {name: i for i, name in enumerate(SUBGOAL_SCHEMA)}

SUBGOAL_DISPLAY = {
    "MOVE_TO_SOURCE_OBJECT": "Move to source object",
    "PICK_SOURCE_OBJECT": "Pick source object",
    "MOVE_SOURCE_TO_TARGET": "Move source to target",
    "POUR_SOURCE_INTO_TARGET": "Pour source into target",
    "MOVE_SOURCE_BACK": "Move source back",
    "RELEASE_SOURCE_OBJECT": "Release source object",
    "MOVE_TO_TOOL_OBJECT": "Move to tool object",
    "PICK_TOOL_OBJECT": "Pick tool object",
    "MOVE_TOOL_TO_TARGET": "Move tool to target",
    "STIR_TARGET": "Stir target",
    "MOVE_TOOL_BACK": "Move tool back",
    "RELEASE_TOOL_OBJECT": "Release tool object",
}

DEFAULT_SUBGOAL_STEPS = {
    0: 180,
    1: 250,
    2: 350,
    3: 250,
    4: 180,
    5: 200,
    6: 200,
    7: 250,
    8: 250,
    9: 250,
    10: 180,
    11: 180,
}

ACTION_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

SYSTEM_INSTRUCTION = """You are an expert robot manipulation planner for a 6-DoF SO-101 robot arm.

Your task is to convert a user's natural-language laboratory manipulation command into the fixed Reference Subgoal Schema below.

You must output exactly the following 12 subgoals in order, using the exact uppercase labels.

[Reference Subgoal Schema]
0. MOVE_TO_SOURCE_OBJECT
1. PICK_SOURCE_OBJECT
2. MOVE_SOURCE_TO_TARGET
3. POUR_SOURCE_INTO_TARGET
4. MOVE_SOURCE_BACK
5. RELEASE_SOURCE_OBJECT
6. MOVE_TO_TOOL_OBJECT
7. PICK_TOOL_OBJECT
8. MOVE_TOOL_TO_TARGET
9. STIR_TARGET
10. MOVE_TOOL_BACK
11. RELEASE_TOOL_OBJECT

[Physical Constraints]
1. The robot can hold only one object at a time.
2. Before picking any object, the robot must first move to that object.
3. The source object must be picked before moving or pouring.
4. After pouring, the source object must be moved back and released before picking the tool.
5. Before stirring, the robot must move to the tool object, pick it, and move it to the target.
6. After stirring, the tool must be moved back and released.
7. Do not invent new subgoal labels.
8. Do not omit any required subgoal.
9. Always include [Subgoals] and then output only a numbered list.

[Output Format]
[Subgoals]
1. MOVE_TO_SOURCE_OBJECT(source=..., target=...)
2. PICK_SOURCE_OBJECT(source=..., target=...)
3. MOVE_SOURCE_TO_TARGET(source=..., target=...)
4. POUR_SOURCE_INTO_TARGET(source=..., target=..., amount=...)
5. MOVE_SOURCE_BACK(source=...)
6. RELEASE_SOURCE_OBJECT(source=...)
7. MOVE_TO_TOOL_OBJECT(tool=...)
8. PICK_TOOL_OBJECT(tool=...)
9. MOVE_TOOL_TO_TARGET(tool=..., target=...)
10. STIR_TARGET(target=..., seconds=...)
11. MOVE_TOOL_BACK(tool=...)
12. RELEASE_TOOL_OBJECT(tool=...)
"""

FEW_SHOT = """### [Few-shot Example]
User: "비커 A 안에 있는 시약을 비커 B에 붓고, 유리 막대로 5초 동안 섞어줘."
Assistant:
[Subgoals]
1. MOVE_TO_SOURCE_OBJECT(source=beaker_A, target=beaker_B)
2. PICK_SOURCE_OBJECT(source=beaker_A, target=beaker_B)
3. MOVE_SOURCE_TO_TARGET(source=beaker_A, target=beaker_B)
4. POUR_SOURCE_INTO_TARGET(source=beaker_A, target=beaker_B, amount=all)
5. MOVE_SOURCE_BACK(source=beaker_A)
6. RELEASE_SOURCE_OBJECT(source=beaker_A)
7. MOVE_TO_TOOL_OBJECT(tool=stick)
8. PICK_TOOL_OBJECT(tool=stick)
9. MOVE_TOOL_TO_TARGET(tool=stick, target=beaker_B)
10. STIR_TARGET(target=beaker_B, seconds=5)
11. MOVE_TOOL_BACK(tool=stick)
12. RELEASE_TOOL_OBJECT(tool=stick)
"""

SUBGOAL_LINE_PATTERN = re.compile(
    r"^\s*\d+\.\s*([A-Z_]+)(?:\((.*?)\))?\s*$",
    re.MULTILINE,
)


class MLPLowLevelPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, dropout=0.05):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def parse_subgoal_arguments(arg_str):
    args = {}
    if not arg_str:
        return args

    parts = [p.strip() for p in arg_str.split(",") if p.strip()]
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            args[k.strip()] = v.strip()

    return args


def parse_subgoals(llm_output):
    section = llm_output.split("[Subgoals]")[-1] if "[Subgoals]" in llm_output else llm_output
    matches = SUBGOAL_LINE_PATTERN.findall(section)

    parsed = []
    for label, arg_str in matches:
        label = label.strip()

        if label not in SUBGOAL_TO_INDEX:
            continue

        parsed.append(
            {
                "label": label,
                "index": SUBGOAL_TO_INDEX[label],
                "arguments": parse_subgoal_arguments(arg_str),
            }
        )

    return parsed


def generate_robot_plan(user_command):
    if client is None:
        print("[ERROR] HF_TOKEN이 설정되어 있지 않습니다.")
        print("Windows: set HF_TOKEN=your_token") ## 자신의 hugging face token 넣기
        print("Mac/Linux: export HF_TOKEN=your_token")
        return []

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {
            "role": "user",
            "content": (
                f"{FEW_SHOT}\n\n"
                f"### [Current Task]\n"
                f'User: "{user_command}"\n'
                f"Assistant:"
            ),
        },
    ]

    try:
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            temperature=0.01,
            max_tokens=2048,
        )

        if not completion.choices:
            print("[WARN] LLM response choices 없음")
            return []

        reasoning = getattr(completion.choices[0].message, "reasoning", None) or ""
        content = completion.choices[0].message.content or ""

        print("\n--- [DEBUG] reasoning ---")
        print(reasoning)
        print("\n--- [DEBUG] content ---")
        print(content)
        print("--- [DEBUG] end ---\n")

        if "[Subgoals]" in content:
            llm_output = content
        elif "[Subgoals]" in reasoning:
            llm_output = reasoning
        else:
            llm_output = content or reasoning

        parsed = parse_subgoals(llm_output)

        if len(parsed) != 12:
            print(f"[WARN] Parsed subgoal length = {len(parsed)}, expected 12")
            return parsed

        expected = list(range(12))
        got = [p["index"] for p in parsed]

        if got != expected:
            print(f"[WARN] Subgoal order mismatch: {got}")
            parsed = sorted(parsed, key=lambda x: x["index"])

        return parsed

    except Exception as e:
        print(f"[ERROR] API 오류: {e}")
        return []


def extract_stir_seconds(parsed_actions, default_seconds=5):
    for act in parsed_actions:
        if act["label"] == "STIR_TARGET":
            args = act.get("arguments", {})
            if "seconds" in args:
                match = re.search(r"\d+", args["seconds"])
                if match:
                    return int(match.group(0))
    return default_seconds


def load_policy(model_path, device):
    ckpt = torch.load(model_path, map_location=device)

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("checkpoint must contain model_state_dict and normalization statistics")

    model = MLPLowLevelPolicy(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt["output_dim"]),
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
        dropout=float(ckpt.get("dropout", 0.05)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    norm_stats = {
        "input_dim": int(ckpt["input_dim"]),
        "state_dim": int(ckpt["state_dim"]),
        "subgoal_dim": int(ckpt["subgoal_dim"]),
        "output_dim": int(ckpt["output_dim"]),
        "input_mean": np.asarray(ckpt["input_mean"], dtype=np.float32).reshape(-1),
        "input_std": np.asarray(ckpt["input_std"], dtype=np.float32).reshape(-1),
        "action_mean": np.asarray(ckpt["action_mean"], dtype=np.float32).reshape(-1),
        "action_std": np.asarray(ckpt["action_std"], dtype=np.float32).reshape(-1),
    }

    if norm_stats["subgoal_dim"] != 12:
        print(f"[WARN] checkpoint subgoal_dim={norm_stats['subgoal_dim']}, expected 12")

    return model, norm_stats


def make_subgoal_onehot(subgoal_idx, subgoal_dim):
    onehot = np.zeros(subgoal_dim, dtype=np.float32)
    if 0 <= subgoal_idx < subgoal_dim:
        onehot[subgoal_idx] = 1.0
    return onehot


def predict_action(model, norm_stats, state_vec, subgoal_idx, device):
    state_vec = np.asarray(state_vec, dtype=np.float32).reshape(-1)

    expected_state_dim = norm_stats["state_dim"]
    if state_vec.shape[0] < expected_state_dim:
        pad = np.zeros(expected_state_dim - state_vec.shape[0], dtype=np.float32)
        state_vec = np.concatenate([state_vec, pad], axis=0)
    elif state_vec.shape[0] > expected_state_dim:
        state_vec = state_vec[:expected_state_dim]

    subgoal_onehot = make_subgoal_onehot(subgoal_idx, norm_stats["subgoal_dim"])

    input_vec = np.concatenate([state_vec, subgoal_onehot], axis=0).astype(np.float32)

    if input_vec.shape[0] != norm_stats["input_dim"]:
        raise ValueError(
            f"Input dimension mismatch: got {input_vec.shape[0]}, expected {norm_stats['input_dim']}"
        )

    input_norm = (input_vec - norm_stats["input_mean"]) / norm_stats["input_std"]

    x = torch.tensor(input_norm, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_norm = model(x).cpu().numpy()[0]

    pred_action = pred_norm * norm_stats["action_std"] + norm_stats["action_mean"]

    return pred_action.astype(np.float32)


def get_joint_state_from_obs(obs):
    return np.array(
        [
            float(obs["shoulder_pan.pos"]),
            float(obs["shoulder_lift.pos"]),
            float(obs["elbow_flex.pos"]),
            float(obs["wrist_flex.pos"]),
            float(obs["wrist_roll.pos"]),
            float(obs["gripper.pos"]),
        ],
        dtype=np.float32,
    )


def build_state_for_inference(joint_state, norm_stats):
    joint_state = np.asarray(joint_state, dtype=np.float32).reshape(-1)

    state_dim = norm_stats["state_dim"]

    if joint_state.shape[0] == state_dim:
        return joint_state

    if joint_state.shape[0] < state_dim:
        vision_dummy = np.zeros(state_dim - joint_state.shape[0], dtype=np.float32)
        return np.concatenate([joint_state, vision_dummy], axis=0)

    return joint_state[:state_dim]


def clip_action(pred_action):
    pred_action = np.asarray(pred_action, dtype=np.float32).copy()

    if pred_action.shape[0] >= 6:
        pred_action[0] = np.clip(pred_action[0], -180.0, 180.0)
        pred_action[1] = np.clip(pred_action[1], -180.0, 180.0)
        pred_action[2] = np.clip(pred_action[2], -180.0, 180.0)
        pred_action[3] = np.clip(pred_action[3], -180.0, 180.0)
        pred_action[4] = np.clip(pred_action[4], -180.0, 180.0)
        pred_action[5] = np.clip(pred_action[5], 0.0, 100.0)

    return pred_action


def run_dry(parsed_actions, model, norm_stats, device, stir_seconds):
    print("\n[DRY-RUN] 가상 제어 루프 시작")

    fps = 30
    current_virtual_joints = np.zeros(6, dtype=np.float32)

    for act in parsed_actions:
        subgoal_idx = act["index"]
        steps_to_run = stir_seconds * fps if subgoal_idx == 9 else DEFAULT_SUBGOAL_STEPS[subgoal_idx]

        print(f"\n▶️ [DRY-RUN Subgoal {subgoal_idx}] {SUBGOAL_DISPLAY[act['label']]}")
        print(f"   args: {act['arguments']}")

        for step in [0, 1, steps_to_run - 1]:
            if step >= steps_to_run:
                continue

            state_vec = build_state_for_inference(current_virtual_joints, norm_stats)

            pred_action = predict_action(
                model=model,
                norm_stats=norm_stats,
                state_vec=state_vec,
                subgoal_idx=subgoal_idx,
                device=device,
            )

            pred_action = clip_action(pred_action)
            current_virtual_joints = pred_action[:6].astype(np.float32)

            print(f"   step {step+1:4d}/{steps_to_run:4d} | action={pred_action.round(3)}")

    print("\n[DRY-RUN SUCCESS] 완료")


def run_robot(parsed_actions, model, norm_stats, device, stir_seconds, port="COM4"):
    if not LEROBOT_AVAILABLE:
        print("[ERROR] LeRobot import 실패. --dry-run 모드로 먼저 확인하세요.")
        return

    robot_cfg = SO100FollowerConfig(port=port)
    robot_cfg.cameras = {}
    robot = make_robot_from_config(robot_cfg)

    print("\n--------------------------------------------------")
    print("[SAFETY] 로봇 시작 전 초기 위치와 주변 장애물을 확인하세요.")
    print("[SAFETY] 중단하려면 Ctrl + C를 누르세요.")
    print("--------------------------------------------------")

    for c in range(5, 0, -1):
        print(f"로봇 작동 {c}초 전...")
        time.sleep(1.0)

    robot.connect()

    try:
        fps = 30
        init_obs = robot.get_observation()
        current_virtual_joints = get_joint_state_from_obs(init_obs)

        for act in parsed_actions:
            subgoal_idx = act["index"]
            steps_to_run = stir_seconds * fps if subgoal_idx == 9 else DEFAULT_SUBGOAL_STEPS[subgoal_idx]

            print("\n==================================================")
            print(f"▶️ [Subgoal {subgoal_idx}] {SUBGOAL_DISPLAY[act['label']]}")
            print(f"args: {act['arguments']}")
            print("==================================================")

            for step in range(steps_to_run):
                start_time = time.perf_counter()

                obs = robot.get_observation()
                joint_state = get_joint_state_from_obs(obs)
                state_vec = build_state_for_inference(joint_state, norm_stats)

                pred_action = predict_action(
                    model=model,
                    norm_stats=norm_stats,
                    state_vec=state_vec,
                    subgoal_idx=subgoal_idx,
                    device=device,
                )

                pred_action = clip_action(pred_action)
                current_virtual_joints = pred_action[:6].astype(np.float32)

                action_dict = {
                    name: float(val)
                    for name, val in zip(ACTION_NAMES, pred_action[:6])
                }

                robot.send_action(action_dict)

                if (step + 1) % 50 == 0 or step == steps_to_run - 1:
                    print(
                        f"   progress: {step+1:4d}/{steps_to_run:4d} "
                        f"({(step+1)/steps_to_run*100:5.1f}%)"
                    )

                dt_s = time.perf_counter() - start_time
                precise_sleep(max(1.0 / fps - dt_s, 0.0))

        print("\n🎉 모든 subgoal sequence 완료")

    except KeyboardInterrupt:
        print("\n[STOP] 사용자 중단")
    finally:
        robot.disconnect()
        print("로봇 연결 해제 완료")


def main():
    print("==================================================")
    print("HSG-BC Integrated Inference System")
    print("==================================================")

    dry_run = "--dry-run" in sys.argv
    if dry_run:
        sys.argv.remove("--dry-run")
        print("[DRY-RUN] 하드웨어 미연결 테스트 모드")

    port = "COM4"
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = sys.argv[idx + 1]
            del sys.argv[idx:idx + 2]

    if len(sys.argv) > 1:
        user_command = " ".join(sys.argv[1:])
    else:
        user_command = input("자연어 명령 입력: ").strip()
        if not user_command:
            user_command = "비커 A에 있는 시약을 비커 B에 붓고, 유리 막대로 5초 동안 섞어줘"

    parsed_actions = generate_robot_plan(user_command)

    if not parsed_actions:
        print("[ERROR] LLM plan 생성 실패")
        return

    if len(parsed_actions) != 12:
        print(f"[ERROR] subgoal 개수 오류: {len(parsed_actions)}개 생성됨. 12개 필요.")
        for i, act in enumerate(parsed_actions, 1):
            print(f"{i}. {act['label']} {act['arguments']}")
        return

    print("\n[Final Subgoal Plan]")
    for i, act in enumerate(parsed_actions, 1):
        print(f"{i:02d}. {SUBGOAL_DISPLAY[act['label']]} | {act['arguments']}")

    stir_seconds = extract_stir_seconds(parsed_actions, default_seconds=5)
    print(f"\n[INFO] Stir duration: {stir_seconds} sec")

    model_path = "mlp_low_level_policy_final.pth"
    if not os.path.exists(model_path):
        model_path = "mlp_low_level_policy.pth"
    if not os.path.exists(model_path):
        model_path = "mlp_policy.pth"

    if not os.path.exists(model_path):
        print(f"[ERROR] 모델 가중치 파일 없음: {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading model: {model_path}")

    model, norm_stats = load_policy(model_path, device)

    print("[SUCCESS] Model loaded")
    print(f"[INFO] input_dim={norm_stats['input_dim']}")
    print(f"[INFO] state_dim={norm_stats['state_dim']}")
    print(f"[INFO] subgoal_dim={norm_stats['subgoal_dim']}")
    print(f"[INFO] output_dim={norm_stats['output_dim']}")

    if dry_run:
        run_dry(parsed_actions, model, norm_stats, device, stir_seconds)
    else:
        run_robot(parsed_actions, model, norm_stats, device, stir_seconds, port=port)


if __name__ == "__main__":
    main()