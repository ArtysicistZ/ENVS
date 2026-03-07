# DeepX cluster SSH access

The main READMEs assume you can already SSH to the cluster (e.g. `ssh kevinzyz@deepx-a100-40g-2`). They do **not** describe how to reach DeepX from your laptop (e.g. via Google Cloud).

---

## Do I need a new window / Google Cloud?

- **If DeepX is only reachable through a jump host (e.g. Google Cloud bastion):**  
  You don’t need to “open a new window, SSH to GCP, then SSH to node 3” manually each time. You can use a **single SSH command** (or one host alias) that goes **laptop → jump host → DeepX node 3** automatically.

- **Shortcut:** put the jump in your SSH config and then run one command (see below).

---

## Shortcut: one command to node 3 (via jump host)

If your path is:

**Laptop → Google Cloud (or other bastion) → DeepX login node → DeepX node 3**

add this to **`~/.ssh/config`** on your laptop (adjust hostnames and key paths):

```ssh-config
# Jump host (e.g. Google Cloud VM or bastion)
Host deepx-jump
    HostName YOUR_JUMP_HOST_IP_OR_NAME
    User YOUR_JUMP_USER
    IdentityFile ~/.ssh/your_jump_key

# DeepX login node (reachable from jump)
Host deepx-login
    HostName deepx-a100-40g-2
    User kevinzyz
    ProxyJump deepx-jump
    IdentityFile ~/.ssh/your_deepx_key

# DeepX node 3 (reachable from login node; use per-node key)
Host deepx-node3
    HostName deepx-a100-40g-3
    User kevinzyz
    ProxyJump deepx-login
    IdentityFile /home/kevinzyz/hansenzuishuai/ssh_keys_for_users/ssh_keys_for_users_172.174.34.71/ssh_keys_for_users/kevinzyz_id_ed25519
```

Then from your **laptop** you can run:

```bash
ssh deepx-node3
```

and you’ll land on node 3 in one step. (If your repo lives elsewhere on the jump/login host, use the path that exists on **deepx-login** for `IdentityFile` when connecting to node 3, or rely on agent forwarding — see below.)

---

## If you’re already on a DeepX node (e.g. node 2)

From **inside** the cluster you don’t need Google Cloud. Use the run script with the per-node key:

```bash
ssh -i ~/hansenzuishuai/ssh_keys_for_users/ssh_keys_for_users_172.174.34.71/ssh_keys_for_users/kevinzyz_id_ed25519 kevinzyz@deepx-a100-40g-3
```

Or run training on node 3 from node 2:

```bash
./scripts/run_remote_env_on_server.sh 3
```

### "Permission denied (publickey)" from node 2 → node 3

Node 3 must have the **public** key for the key in `ssh_keys_for_users/.../kevinzyz_id_ed25519` in `~/.ssh/authorized_keys`. If your **home directory is shared** across nodes (NFS), you can add it from node 2 (same `~`):

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMe4dvGTX90uf054u9SsKCcSJ/jZGn/k6EBbU7arO4s+ kevinzyz@deepx-a100-40g-2' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

Then try again: `./scripts/run_remote_env_on_server.sh 3`. If home is not shared, you’ll need to get a shell on node 3 another way (e.g. cluster console or admin) and run the same `echo ... >> ~/.ssh/authorized_keys` there.

---

## Optional: SSH agent forwarding

If the key for node 3 only exists on the jump/login host (not on your laptop), you can forward the agent so the key on the intermediate host is used:

```ssh-config
Host deepx-login
    HostName deepx-a100-40g-2
    User kevinzyz
    ProxyJump deepx-jump
    ForwardAgent yes
```

Then from your laptop: `ssh deepx-login`, and from there `ssh -i .../kevinzyz_id_ed25519 kevinzyz@deepx-a100-40g-3`. The `IdentityFile` for node 3 in the config above only works if that path exists on the machine that actually connects to node 3 (the login node). So either put the key on the login node at that path, or use `ForwardAgent yes` and load the key into the agent on the login node.

---

## Summary

| Question | Answer |
|----------|--------|
| Do READMEs say to use Google Cloud to reach DeepX? | No. They assume you can already SSH to the cluster. |
| Do I need to open a new window and SSH twice? | No. Use `~/.ssh/config` with `ProxyJump` so one `ssh deepx-node3` goes laptop → jump → node 3. |
| Shortcut from laptop | Add `Host deepx-node3` with `ProxyJump deepx-login` (and `deepx-login` with `ProxyJump deepx-jump`), then `ssh deepx-node3`. |

Fill in `YOUR_JUMP_HOST_IP_OR_NAME`, `YOUR_JUMP_USER`, and key paths to match your actual Google Cloud / DeepX setup.
