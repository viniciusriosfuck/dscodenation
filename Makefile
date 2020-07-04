script=main.py

run:
	@python data-science-0/$(script)

build:
	@python setup.py build

sdist:
	@python setup.py sdist

install:
	@python setup.py install

.PHONY: clean	
clean:
	rm -fR build dist *egg-info