#!/bin/bash

_make() (
	set -e
	set -u

	# Config
	_name=ipy-slurm-exec
	_ver=`cat VERSION`

	_live_mode=false
	_index_url=https://test.pypi.org/simple
	if [ "${1:-}" = "--live" ]; then
		_live_mode=true
		_index_url=https://pypi.org/simple
	fi

	rm -f setup.cfg
	cat setup.cfg.template | sed "s/<VERSION>/$_ver/g" > setup.cfg
	_nameu=`echo "$_name" | sed "s/-/_/g"`
	_outdir="dist/$_ver"

	# Cleanup last build
	if [ -d "$_nameu.egg-info" ]; then
		rm -r "$_nameu.egg-info"
	fi
	if [ -d src/"$_nameu.egg-info" ]; then
		rm -r src/"$_nameu.egg-info"
	fi
	if [ -d "$_outdir" ]; then rm -r "$_outdir" ; fi
	mkdir -p "$_outdir"

	# Build wheel
	mkdir -p ~/venvs
	if [ ! -e ~/venvs/publish ]; then
	  python -m venv ~/venvs/publish
	fi
	source ~/venvs/publish/bin/activate
	python -m pip install --upgrade pip setuptools wheel build twine
	python -m build --outdir "$_outdir"
	deactivate

	# Publish
	if $_live_mode ; then
		python -m twine upload "$_outdir"/*
	else
		python -m twine upload --repository testpypi "$_outdir"/*
	fi

	# Note: to download from Test PYPI successfully,
	# need to download dependencies from PYPI like this:
	rm -rf ~/venvs/tmp
	python -m venv ~/venvs/tmp
	source ~/venvs/tmp/bin/activate
	pip install --no-deps --index-url "$_index_url" ipy-slurm-exec
	_reqs=`pip show ipy-slurm-exec | grep "^Requires:" | cut -d':' -f2 | sed 's/,//g'`
	if [ -n "$_reqs" ]; then
		pip install -v $_reqs
	fi
	deactivate
)

_make "$@"
