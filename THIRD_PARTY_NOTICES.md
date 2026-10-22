# Third-party notices

## Delphi

AEGIS follows the design of Delphi (Shmatko et al., "Learning the natural history of human disease with generative
transformers", Nature 2025; https://github.com/gerstung-lab/Delphi). The following parts are adapted from the Delphi
code: the model architecture and the competing-exponentials loss with its smoothing constant (`aegis/model.py`), the
no-event tokens and the random shift of the lifestyle tokens during training (`aegis/data.py`). AEGIS contains no
Delphi model weights and no data.

Delphi's code is released under the MIT licence:

```
MIT License

Copyright (c) 2024 Gerstung Lab

MIT license applies only to code; the model weights use CC BY-NC-ND 4.0 instead.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Libraries

PyTorch (BSD-3-Clause), Opacus (Apache-2.0), NumPy, pandas and SciPy (BSD-3-Clause) are used as installed
dependencies and are not redistributed.
