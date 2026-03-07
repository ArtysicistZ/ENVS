# Remote env: trainer flow and server logs

## AWS configuration reference

When the remote env runs on AWS (OSWorld VMs / EC2), use this as the reference for networking and env vars.

- **Security group for OSWorld VMs** (`AWS_SECURITY_GROUP_ID`): Inbound — SSH 22, HTTP 80, Custom TCP 5000 (OSWorld backend), 5910 (NoVNC), 8006 (VNC), 8080 (VLC), 8081, 9222 (Chrome); sources per your VPC (e.g. 172.31.0.0/16 for internal, 0.0.0.0/0 for SSH/5910). Outbound: all traffic to 0.0.0.0/0.
- **Host security group:** Open port **15001** so the training cluster can reach the remote env server; optionally 8080 for a monitor.
- **VPC:** Host and all client VMs in the **same subnet**. Set `AWS_SUBNET_ID` (and optionally `AWS_REGION`, e.g. `us-east-1`).
- **Env vars:** `AWS_REGION`, `AWS_SUBNET_ID`, `AWS_SECURITY_GROUP_ID`; for instance launch: `KEY_NAME`, `INSTANCE_TYPE`, `IMAGE_ID_MAP` (e.g. `us-east-1`: official OSWorld AMI). Configure AWS CLI with `aws configure` (Access Key, Secret Key, default region).

---

## Training-side checklist (AWS EC2 remote server)

- **Config key:** Use **`env.remote_server_url`** (not `remote_env_url`). Example: `remote_server_url: "http://18.206.172.166:15001"`.
- **Config file:** e.g. `configs/smoke_remote_env.yaml`; run with `python -m verl.trainer.main config=configs/smoke_remote_env.yaml`.
- **Connectivity:** EC2 security group must allow **inbound TCP 15001** from the training cluster. Verify from the cluster: `curl http://<EC2_IP>:15001/health` or `./scripts/verify_remote_env_connection.sh`.
- **Server:** On EC2, start with `./scripts/start_remote_env_aws.sh` (or run uvicorn with `PROVIDER=aws` and required AWS env vars). See `docs/REMOTE_ENV_SERVER_RUNBOOK.md`. The server loads `.env` at startup so `OPENAI_API_KEY` (and AWS vars) are available for tasks; the start script in [arpo_remote_env](https://github.com/hanshengzhu0001/arpo_remote_env) also sources `.env` before starting.

---

## Trainer order (reset → step → evaluate)

The trainer **does** perform reset, then steps, then evaluate. Exact flow in `verl/trainer/ray_trainer.py`:

1. **Reset**  
   - `start_reset_envs(batch_dict)` (line ~843) builds Ray futures: `worker.reset.remote(task_config)` per env.  
   - `reset_outputs = ray.get(reset_envs_object)` (line ~869) waits for all resets.  
   - So every episode starts with a **reset** on each env.

2. **Step loop**  
   - `for step_idx in range(self.config.env.max_steps)` (line ~875):  
     - `prepare_vllm_inputs_full(env_outputs)` → build VLM batch from current obs.  
     - `generate_sequences` → model predicts actions.  
     - `worker.step.remote(action_text)` → env step; `env_outputs = ray.get(futures)`.  
   - When a step returns `is_done=True` for an env, that env’s **evaluate** is fired (async):  
     - `eval_results_objects[cur_env_idx] = self.env_workers[cur_env_idx].evaluate.remote()` (line ~946).  
   - Loop breaks when `is_all_done` (all envs reported done).

3. **Evaluate**  
   - After the step loop: `eval_results = ray.get(eval_results_objects)` (line ~968).  
   - So **evaluate is only called for envs that have reported `is_done=True`** at least once (typically after one or more steps). If reset fails and we break with `batch_skipped`, we never call evaluate and use `eval_results = [0.0] * len(task_configs)`.

So the order is: **reset → step (0..max_steps) → evaluate** (only for envs that are done). Evaluate is not called before any step; it is called after the env has been stepped at least until `is_done` is True.

---

## What to look for in remote env server logs

Run the server with stdout visible (e.g. `python scripts/remote_env_server.py` or via your process manager). During a full training run you should see:

| Event | Log line(s) |
|-------|-------------|
| **Reset requested** | (No single “reset received” line; first evidence is success/failure below.) |
| **Reset OK** | `Reset OK: VM screenshot obtained (... bytes), returning obs_messages with image. Instruction: ...` |
| **Reset failed** | `Env reset exception: ...` or `Reset: screenshot is None (VM/container not ready...). Returning obs_messages=None.` |
| **Step** | `step_parse: ... actions=[...] format_reward=... pred_preview=...` (once per step) |
| **Evaluate requested** | `POST /env/evaluate received (instruction=..., step_counter=...)` |
| **Evaluate OK** | `Evaluation completed: score=..., instruction=..., step_counter=...` |
| **Evaluate skipped (503)** | `Evaluation skipped: env not fully started (no setup_controller)...` or `...env has no setup_controller (reset failed)...` |
| **Evaluate error (0.0)** | `Evaluation error: ...` and traceback |

If you see **no** `POST /env/evaluate received` during a run, the cluster is not reaching `/env/evaluate` (e.g. connection/timeout or trainer never got `is_done=True`). If you see that line but then `Evaluation skipped` (503) or `Evaluation error`, the server is returning 503 or 0.0 and the client will record eval 0.

---

## Data in use: task configs, screenshots, Chrome, API

Training uses task data and the remote env as follows:

1. **Data**  
   - `data.train_files` (e.g. `OSWorld/evaluation_examples/test_smoke_4.json`) lists task IDs per domain (e.g. `chrome`).  
   - For each task, the trainer loads `examples/<domain>/<id>.json`, which contains `instruction`, `config` (Chrome launch, etc.), and `evaluator`.  
   - That full **task_config** is sent in `POST /env/reset` so the server can start the VM, launch Chrome, and return a screenshot.

2. **Remote server**  
   - The server must be able to: start (or attach to) a VM, run Chrome (or the app under test), capture a **screenshot**, and expose **reset / step / evaluate** over HTTP.  
   - If reset returns **503** or **obs_messages=None**, the trainer sees no screenshot and will skip the batch; fix the server (VM ready, `setup_controller`, Chrome, screenshot) so reset returns 200 with `obs_messages` containing an image.

3. **Verify end-to-end**  
   - From the project root (or with correct `config` path):  
     `python scripts/verify_data_and_remote_env.py [configs/smoke_remote_env.yaml]`  
   - This checks: task list and `examples/` files exist, one full task_config has `instruction` and `config`, then calls `POST /env/reset` and asserts the response includes an image (screenshot).  
   - Use `--skip-reset` to only verify data loading.  
   - If this script passes, data is in use and screenshot/Chrome/API are working.

---

## Quick checks

- **Cluster → server:** From a cluster node: `curl -s http://<EC2_IP>:15001/health` and `python scripts/test_evaluate_endpoint.py [BASE_URL]`.  
- **Data + reset (screenshot):** `python scripts/verify_data_and_remote_env.py configs/smoke_remote_env.yaml`.  
- **Server logs:** Tail the process that runs `remote_env_server.py` while training; confirm `Reset OK`, `step_parse`, and `POST /env/evaluate received` / `Evaluation completed` (or the 503/error lines above).
