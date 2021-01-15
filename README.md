# Usage
For general usage information, see `intervention-learning --help`.

To collect imitation-learning data from an expert driver, first download the expert model from [the Learning by Cheating publication](https://github.com/dotchen/LearningByCheating):

```shell
$ wget http://www.cs.utexas.edu/~dchen/lbc_release/ckpts/privileged/model-128.th
```

Run the Carla simulator. Then run:

```shell
$ intervention-learning collect-teacher-examples \
    --teacher-checkpoint ./model-128.th \
    --directory ./a-new-dataset \
    --num-episodes 5
```

# Setup
## System dependencies
There are some system dependencies.
You can use Conda `environment.yml` provided in the [intervention-scripts repository](https://github.com/beskhue/intervention-scripts) to help you get started.
Alternatively, the current repository provides `./flake.nix` and `./shell.nix` to prepare a development environment using [Nix](https://nixos.org).

```shell
# On a flake-enabled Nix system:
$ nix develop
# On a non-flake-enabled Nix system:
$ nix-shell
```

## Setup
A `requirements.txt` is provided to pin dependencies to some quite specific versions known to work well.

```shell
$ https://github.com/tomcur/intervention.git
$ cd intervention
$ python3 -m pip install -r requirements.txt
$ python3 -m pip install -e .
```

### Carla
You will need to have Carla's Python client API package installed, matching the Carla version you'll be using.
To install using the Carla-provided eggs, perform e.g.:

```shell
$ wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.1_RSS.tar.gz
$ mkdir CARLA_0.9.10.1_RSS
$ cd CARLA_0.9.10.1_RSS
$ tar -xf ../CARLA_0.9.10.1_RSS.tar.gz
$ easy_install ./PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
```

### Platform-specific requirements
Note that you might need to override some dependencies (probably PyTorch and Torchvision) to satisfy your platform's requirements.
For example, to install pytorch and torchvision with CUDA 10.1 support, apply the following patch

```diff
--- a/requirements.txt
+++ b/requirements.txt
@@ -2,2 +2,2 @@
-torch==1.7.1
+torch==1.7.1+cu101
-torchvision==0.8.2
+torchvision==0.8.2+cu101
```

and install requirements with

```sh
 $ pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

For more information see [PyTorch's documentation on this matter](https://pytorch.org/get-started/previous-versions/).
