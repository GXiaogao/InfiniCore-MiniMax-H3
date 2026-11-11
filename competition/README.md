# MiniMax H3 preliminary source branch

This branch keeps upstream history and preserves the archived source changes: Hygon InfiniOp-composed MHA, gated-residual API/binding plus NineToothed host adapter, singleton layout support and the no-CPU context guard. `source-provenance.json` records its upstream and fixed archive. Third-party submodules remain at their upstream Git links; no competitor changes were found in their archived tracked source files. Bundled CI-only files and Windows executable-bit drift are not treated as numerical modifications.

`ninetoothed-gated-v3/` contains the actual Nine DSL and original gfx906 exporter/host glue. This historical path passed single-card correctness but device-stream performance regressed; it is not automatically enabled for the gfx936 full model. The formal H3 gated switch stays0 by default. Do not claim eight-card Nine speedup from this branch.

Build in the existing gfx936 vendor environment, in a writable job workspace. Set `IFAB_AFS_HOME` and source `$IFAB_AFS_HOME/minimax-h3/setup_infinifabric_env.sh` first. Run `git submodule update --init third_party/spdlog` in this checkout, then run each profile as a separate28-minute job:

```bash
python3 /absolute/InfiniCore-MiniMax-H3/competition/build_from_git.py native
python3 /absolute/InfiniCore-MiniMax-H3/competition/build_from_git.py python
```

The adapter reads committed Git source, verifies/export required submodules, uses current workspace for compilation, existing offline xmake recipe/pybind archives under work/packages, and writes only the new `.infini-public-20261111` prefix and `python-envs/public-git-20261111/site-packages`. Logs go to work/logs/public-git-*. It neither installs into the shared interpreter nor replaces `.infini`. Xmake build-file transformations are build-only and confined to temporary copies. Compiler:vendorDTK; native flags mirror the historical Hygon-only builder, custom generic Nine build option remainsoff. The new Git packaging/build adapter needs a server rebuild; source equality is not a claim that a newly built binary or model passed.
