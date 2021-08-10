This is a tiny reader for CORSIKA-IACT output
---------------------------------------------

Default CORSIKA with IACT-option writes Cherenkov-output into ```eventio``` files.
Here, only the ```C++``` based ```merlict-development-kit``` is able to read ```eventio```.

The workflow is:

- run CORSIKA-IACT and write an eventio-file.
- run merlict to convert ```eventio``` to ```simpleio```.
- Use this reader to read the ```simpleio```.

Simpleio is the same content as eventio, but the individual container are dumped into the file-system. So a corsika-run will be a directory with files for run-header, event-headers, and Cherenkov-photons.

