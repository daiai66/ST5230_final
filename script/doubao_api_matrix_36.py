import json
import os
import time
from datetime import datetime
from typing import Dict, List

from doubao_api_test import (
    DILEMMAS,
    RESPONSIBILITY,
    ROLES,
    DoubaoAPIClient,
    build_system_prompt,
    build_user_prompt,
)


def run_matrix_experiment(
    num_samples_per_cell: int = 40,
    dilemma_index: int = 0,
    resp_key: str = "neutral",
    inter_call_delay: float = 2.0,
    output_prefix: str = "doubao_matrix",
) -> List[Dict]:
    print("=" * 90)
    print("Doubao Matrix Experiment (Responsibility x Role x Samples)")
    print("=" * 90)

    try:
        client = DoubaoAPIClient()
        print("API client initialized")
    except ValueError as exc:
        print(f"Initialization failed: {exc}")
        return []

    if dilemma_index < 0 or dilemma_index >= len(DILEMMAS):
        print(f"Invalid dilemma_index={dilemma_index}, fallback to 0")
        dilemma_index = 0

    selected_dilemma = DILEMMAS[dilemma_index]
    selected_resp_keys = [resp_key]
    total_calls = len(selected_resp_keys) * len(ROLES) * num_samples_per_cell

    print(f"Model: {client.model}")
    print(f"Dilemma: {selected_dilemma['name']}")
    print(
        f"Plan: {len(selected_resp_keys)} responsibility levels x {len(ROLES)} roles x {num_samples_per_cell} samples = {total_calls}"
    )
    print("-" * 90)

    results: List[Dict] = []
    call_index = 0

    for resp_key in selected_resp_keys:
        responsibility = RESPONSIBILITY[resp_key]
        for role_key, role in ROLES.items():
            system_prompt = build_system_prompt(responsibility, role)
            user_prompt = build_user_prompt(responsibility, role, selected_dilemma)

            print(f"\nCell: {resp_key} x {role_key}")
            print(f"System prompt len: {len(system_prompt)} | User prompt len: {len(user_prompt)}")

            for sample_idx in range(1, num_samples_per_cell + 1):
                call_index += 1
                print(f"  [{call_index}/{total_calls}] sample {sample_idx}/{num_samples_per_cell} ...", end=" ")

                response = client.call_doubao(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.7,
                    max_tokens=3000,
                )

                record = {
                    "call_index": call_index,
                    "sample_num": sample_idx,
                    "responsibility": resp_key,
                    "role": role_key,
                    "dilemma": selected_dilemma["id"],
                    "responsibility_name": responsibility["name"],
                    "role_name": role["name"],
                    **response,
                }
                results.append(record)

                if response.get("status") == "success":
                    elapsed = response.get("elapsed_time", 0)
                    tokens = response.get("tokens_used", 0)
                    finish_reason = response.get("finish_reason", "unknown")
                    print(f"OK ({elapsed:.2f}s, tokens={tokens}, finish={finish_reason})")
                else:
                    print(f"FAILED ({response.get('error', 'unknown error')})")

                if inter_call_delay > 0 and call_index < total_calls:
                    time.sleep(inter_call_delay)

    output_dir = os.path.join("dataset", "responses", "raw")
    os.makedirs(output_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{output_prefix}_{total_calls}_{ts}.json")

    with open(output_file, "w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=2, ensure_ascii=False)

    success_count = sum(1 for item in results if item.get("status") == "success")
    error_count = len(results) - success_count
    total_tokens = sum(int(item.get("tokens_used", 0) or 0) for item in results)

    print("\n" + "=" * 90)
    print("Experiment completed")
    print(f"Output file: {output_file}")
    print(f"Records: {len(results)} (expected {total_calls})")
    print(f"Success: {success_count} | Failed: {error_count} | Total tokens: {total_tokens}")
    print("=" * 90)

    return results


if __name__ == "__main__":
    for dilemma_idx in range(len(DILEMMAS)):
        dilemma_id = DILEMMAS[dilemma_idx]["id"]
        for rk in RESPONSIBILITY:
            print(f"\nRunning: {rk} x all 7 roles x 40 samples — {dilemma_id}")
            run_matrix_experiment(
                num_samples_per_cell=40,
                dilemma_index=dilemma_idx,
                resp_key=rk,
                output_prefix=f"doubao_matrix_{rk}_{dilemma_id}",
            )

