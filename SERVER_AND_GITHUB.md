# Put the package on the server and push it to GitHub

The GitHub repository already exists at:

```text
https://github.com/BfsorDfs6/AgeSafer
```

## 1. Upload and unpack on the server

Upload `AgeSafer_ML1M_GMF_reference.zip` to the server, then run:

```bash
mkdir -p /data/fyx/projects/0hjr_project/final
cd /data/fyx/projects/0hjr_project/final
unzip -o /path/to/AgeSafer_ML1M_GMF_reference.zip
```

## 2. Copy into the existing GitHub repository clone

Because the remote repository already contains a README commit, cloning first is
safer than creating an unrelated local history:

```bash
cd /data/fyx/projects/0hjr_project/final
git clone https://github.com/BfsorDfs6/AgeSafer.git AgeSafer_repo
rsync -av --delete \
  AgeSafer_ML1M_GMF_reference/ \
  AgeSafer_repo/ \
  --exclude .git
cd AgeSafer_repo
```

Review files before pushing:

```bash
find . -maxdepth 3 -type f | sort
git status
```

## 3. Commit and push

```bash
git add .
git commit -m "Release lightweight ML-1M + GMF reference implementation"
git push origin main
```

GitHub may require a personal access token or an SSH remote. For SSH:

```bash
git remote set-url origin git@github.com:BfsorDfs6/AgeSafer.git
git push origin main
```

Alternatively, after copying the package into a clean repository directory, run:

```bash
bash scripts/publish_to_github.sh https://github.com/BfsorDfs6/AgeSafer.git main
```

The helper does not store credentials and cannot bypass GitHub authentication.
