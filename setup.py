from setuptools import setup

README = open("README.md").read()

setup(
    author="Andrew Owenson",
    author_email="andrew.owenson@imm.ox.a.cuk",
    description="Jupyter Notebook + Slurm intergration",
    long_description_content_type="text/markdown",
    long_description=README,
    name="ipy-slurm-exec",
    license="Proprietary",
    license_files=["LICENSE.txt"],
    py_modules=["ipy_slurm_exec"],
    install_requires=["ipython"],
    url="https://github.com/WIMM-IT/ipy-slurm-exec",
    version="0.3.1",
)
