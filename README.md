# Installing dependencies
A `requirements.txt` is provided to pin dependencies to some quite specific versions known to work well.
Note that you might need to override some dependencies (probably PyTorch and Torchvision) to satisfy your platform's requirements.
For example, to install pytorch and torchvision with CUDA 10.1 support, apply the following patch

```diff
--- a/requirements.txt
+++ b/requirements.txt
@@ -5,9 +5,9 @@
-torch~=1.5.1
+torch==1.5.1+cu101

-torchvision~=0.6.1
+torchvision==0.6.1+cu101
 ```

and install requirements with

```sh
 $ pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

For more information see [PyTorch's documentation on this matter](https://pytorch.org/get-started/previous-versions/).
