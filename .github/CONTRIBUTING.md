### Create a Pull Request

1. Fork the [repository](https://github.com/THU-BPM/MarkDiffusion) by clicking on the [Fork](https://github.com/THU-BPM/MarkDiffusion/fork) button on the repository's page. This creates a copy of the code under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

```bash
# Replace [username] with your GitHub username
git clone git@github.com:[username]/MarkDiffusion.git
cd MarkDiffusion

# Add the original repository as "upstream" to keep your fork synced
git remote add upstream https://github.com/THU-BPM/MarkDiffusion.git
````

3.  Create a new branch to hold your development changes:

<!-- end list -->

```bash
# It is good practice to sync with upstream before creating a branch
git fetch upstream
git checkout -b dev_your_branch upstream/main
```

4.  Set up a development environment. We recommend using a virtual environment:

<!-- end list -->

```bash
# Install requirements
pip install -r requirements.txt

# Install MarkDiffusion in editable mode
pip install -e .
```

5.  Check code before commit:

Please ensure your code passes local tests and follows the project's coding style.

6.  Submit changes:

<!-- end list -->

```bash
git add .
git commit -m "feat: add awesome feature"

# Sync with upstream again to avoid conflicts
git fetch upstream
git rebase upstream/main

# Push to your own fork (origin)
git push -u origin dev_your_branch
```

7.  Create a Pull Request from your branch `dev_your_branch` at [your fork page](https://www.google.com/search?q=https://github.com/THU-BPM/MarkDiffusion/pulls) (GitHub will prompt you to merge into THU-BPM).

<!-- end list -->
