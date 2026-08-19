# Access Policy

This file defines the filesystem access boundary for this project.

The agent must treat this file as authoritative for file reading, listing, searching, grepping, finding, summarizing, writing, moving, deleting, and modifying.

## Project Workspace

The current VSCode/Codex workspace is the primary project root.

The project root is represented by:

```text
.
```

All files and directories recursively contained under the current project root are writable.

## Allowed Read/Search Roots

The agent may read, list, inspect, grep, find, search, or summarize files only under:

- .
- /data/dataset
- /home/hyunjin/.cache/huggingface/datasets
- /home/hyunjin/.cache/huggingface/hub
- /home/hyunjin/.agents/skills

Optional project-specific external read roots may be added here if needed:

- <ADD_PROJECT_SPECIFIC_READ_ROOT_HERE>

## Allowed Write Roots

The agent may write, create, modify, move, rename, or delete any file or directory recursively under:

- .

The following approved external write root is also allowed:

- /data/dataset

This means the entire current project workspace is writable, including existing and newly created:

- source code,
- configuration files,
- documentation,
- hidden project files and directories,
- infrastructure files,
- scripts,
- experiment outputs,
- state and workspace files.

Writing outside the current project root remains denied unless an external path is explicitly listed in this file or the user grants permission in the current chat.

## Read-Only External Roots

The following external roots are read-only:

- /home/hyunjin/.cache/huggingface/datasets
- /home/hyunjin/.cache/huggingface/hub
- /home/hyunjin/.agents/skills
- /home/hyunjin/.agents/skills

The agent must not write, delete, move, rename, clean, or modify anything under these read-only external roots.

Important:

- `/data/dataset` is both an allowed read/search root and the approved external write root for missing dataset downloads.
- The Hugging Face cache paths under `/home/hyunjin/.cache/huggingface/` are read-only lookup roots.
- `/home/hyunjin/.agents/skills` is a read-only exception for globally installed Codex skills. The agent may inspect only the skill metadata or selected `SKILL.md` files needed for the current task.
- Do not use or create a project-local dataset cache unless the user explicitly requests it.

## Dataset Cache Policy

Before downloading any dataset, check these roots first, in this exact order:

1. /data/dataset
2. /home/hyunjin/.cache/huggingface/datasets

If the dataset is missing from both locations, download it under a dataset-specific subdirectory of:

```text
/data/dataset
```

For Hugging Face datasets, use this default cache directory:

```text
/data/dataset/huggingface/datasets
```

Do not place dataset contents loosely in the `/data/dataset` root.

Do not write to:

```text
/home/hyunjin/.cache/huggingface/datasets
```

unless the user explicitly changes this policy.

## Deny-by-Default Rule

Everything outside the allowed roots is denied by default.

The agent must not read, list, inspect, grep, find, search, or summarize files outside the allowed read/search roots.

The agent must not write, create, modify, move, delete, or rename files outside the allowed write roots.

## Sensitive Paths

Never access these paths unless the user explicitly grants permission in the current chat:

- /
- /home
- /home/*
- /data/projects outside the current workspace
- other users' home directories
- other users' project directories
- unrelated repositories
- ~/.ssh
- ~/.aws
- ~/.config
- ~/.cache except explicitly allowed Hugging Face cache paths
- shell history files
- credential files
- private keys
- token files
- environment files outside the current workspace

The exact path `/home/hyunjin/.agents/skills` is an explicit read-only exception to the broad `/home` restriction above. No other path under `/home` is allowed unless separately listed or explicitly approved.

A sensitive path located inside the current project workspace is writable under this policy, but the agent should still avoid inspecting or exposing secrets unless the task requires it.

## If Access Is Needed Outside the Policy

If a task appears to require access outside this policy, stop and ask the user.

Do not inspect the external path before permission is granted.

Do not run `ls`, `find`, `grep`, `rg`, `cat`, `head`, `tail`, or Python file reads on denied paths.

## Command Safety Rule

Before running any command that reads or searches files, ensure all target paths are inside the allowed read/search roots.

Before running any command that writes or modifies files, ensure all target paths are inside the allowed write roots.

Commands targeting `.` or paths resolved beneath the current project root are allowed for reading and writing.

## Broad Search Ban

Do not run broad filesystem searches such as:

```bash
find /
find /home
find /data/projects
grep -R PATTERN /
rg PATTERN /
ls /
ls /home
```

unless the user explicitly grants permission.

Project-local searches such as the following are allowed:

```bash
find .
rg PATTERN .
grep -R PATTERN .
```
