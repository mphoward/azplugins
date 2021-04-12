Installation
============

Instructions

Check hoomd-blue and azplugins installation
-------------------------------------------

Follow the hoomd-blue installation guide for compiling it from source and install azplugins either as external or
internal plugin. Make sure you are using compatible versions of hoomd-blue for azplugins.  Execute all hoomd-blue
unit tests with ctest to check the installation, as well as all azplugin unit tests. If azplugins was installed
internally, they will be run at the same time as the hoomd-blue tests, but they can be run seperately by using
``ctest -R azplugins-*`` in the hoomd-blue build folder. If azplugins was installed as external plugin,
``make test`` can be used to execute all unit tests, or ``ctest -R azplugins-*`` in the azplugins build folder.
Every time hoomd-blue or azplugins are installed  on a new machine, or with new compilers, or with changed
prerequisites, all unit tests should be re-run to make sure nothing is broken.

For using hoomd-blue and azplugins as an internal plugin, either open a interactive session of
python3 or execute a script with:

.. code-block:: python

    import hoomd
    from hoomd import azplugins
    hoomd.context.initialize()


For using azplugins when compiled externally:

.. code-block:: python

    import hoomd
    import azplugins
    hoomd.context.initialize()


This should result in no error messages and hoomd-blue will print some useful information (like its version, where it
is running (CPU/GPU), and what compilers were used, when it was compiled) when the simulation context is initialized.
This is very helpful and should be carefully checked to make sure everything is running like intended.  It should match
the information from cmake when azplugins was compiled.

Common pitfalls:

*   The use of inconsistent python3 installations or versions. Hoomd-blue and azplugins need to be compiled with the exact
    same python3 environment, which is the one simulation scripts will be executed with. Additional packages like numpy etc.
    also need to be installed for this python3. When  executing  ``cmake`` for either azplugins or hoomd-blue, the python3
    executable can be specified with  ``-DPYTHON_EXECUTABLE=<path-to-python3>`` if the desired one is not found automatically.
    Since there are many python environment/package managers like pip, anaconda, pyenv ... a careful check of the cluster/local
    system might be needed.

*   Even if the scripts are executed with the correct python3,  one still needs to indicate to python3 where to find both
    hoomd-blue and azplugins. This can be done in two different ways:

        * Appending/prepending the right location to your $PYTHONPATH variable in the ``~/.bash_rc`` or ``~/.profile`` file.
          Usually a new line like this  ``PYTHONPATH="${PYTHONPATH}:path-to-hoomd`` (and analogous for azplugins) needs
          to be added. After editing ``~/.bash_rc`` or ``~/.profile``, the file needs to be sourced ``source ~/.bash_rc``
          or the terminal needs to be restarted for the changes to take effect. Adding the location of hoomd-blue to the
          ``PYTHONPATH`` once installed has the benefit that azplugins will find hoomd-blue automatically when compiled
          externally.

        * Importing the package sys at the very beginning of your script and setting the paths with
          ``sys.path.insert(0, <path-to-hoomd>)``, where ``<path-to-hoomd>`` is the installation location  set with
          ``-DCMAKE_INSTALL_PREFIX=<path-to-hoomd>``. Azplugins are included in the sys path in the same way.
          This is useful if different projects are running different versions of hoomd-blue, because the script explicitly
          states which version is used and multiple versions can be installed in parallel:

          .. code-block:: python

              import sys
              sys.path.insert(0,'~/Programs/hoomd-2.6.0/') # -DCMAKE_INSTALL_PREFIX='~/Programs/hoomd-2.6.0/'
              sys.path.insert(0,'~/Programs/azplugins-0.7.0/') # -DCMAKE_INSTALL_PREFIX='~/Programs/azplugins-2.7.0/'
              import hoomd
              import azplugins
