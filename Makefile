# Conda dev environment helpers
install_conda_reqs:
	conda env create -f dev_environment.yml
conda-activate:
	conda activate omnifit-dev
# Tox tests and builds
test:
	tox -e test
codestyle:
	tox -e codestyle
docs:
	tox -e build_docs
# Build
build: test codestyle
	python -m build --sdist --wheel --outdir dist/
