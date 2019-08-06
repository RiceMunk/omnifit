flake:
	flake8 omnifit --count --show-source --statistics

update-submodules:
	git submodule foreach git pull origin master

clean:
	./setup.py clean

build:
	./setup.py build

test:
	./setup.py test