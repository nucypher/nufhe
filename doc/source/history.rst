---------------
Version history
---------------


0.0.2 (current development version)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **CHANGED:** a ``PerformanceParameters`` object needs to be specialized for the device used (by calling its ``for_device()`` method) before passing it to gates.

* FIXED: a bug in ``tlwe_noiseless_trivial()`` occasionally leading to memory corruption.

* FIXED: a bug where ``PerformanceParameters`` and ``PerformanceParametersForDevice`` objects did not have a correct equality implementation, leading to unnecessary re-compilation of kernels.


0.0.1 (12 Oct 2018)
~~~~~~~~~~~~~~~~~~~

Initial version.
