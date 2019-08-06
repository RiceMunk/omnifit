update-submodules:
	git submodule foreach git pull origin master

clean:
	./setup.py clean

build:
	./setup.py build

test:
	./setup.py test