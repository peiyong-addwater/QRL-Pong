# QRL-Pong

## Setupt the Environment

1. Install Poetry with pipx: `pipx install poetry`
2. Clone the repository
3. Change directory to the repository
4. Run `poetry install` to install the dependencies


- Note: if pytorch import error

  ```
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    File "/home/paperspace/.cache/pypoetry/virtualenvs/qrl-pong-_LKev65F-py3.12/lib/python3.12/site-packages/torch/__init__.py", line 367, in <module>
      from torch._C import *  # noqa: F403
      ^^^^^^^^^^^^^^^^^^^^^^
  ImportError: /home/paperspace/.cache/pypoetry/virtualenvs/qrl-pong-_LKev65F-py3.12/lib/python3.12/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12
  ```
  occurs, then run the following command to fix it:
  ```bash
  export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib')"):$LD_LIBRARY_PATH
  ```
  within poetry shell.

- Note 2: if `poetry install` or `poetry add` pending and block by keyring, then run the following command to fix it:
  ```bash
  export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
  ```
  or
  ```bash
  poetry config keyring.enabled false
  ```

- Note 3: To enable poetry python: `eval $(poetry env activate)`
