.PHONY: all bootstrap runtime clean

all: bootstrap runtime

bootstrap:
	python bootstrap/bootstrap_assembler.py assembler/somasc.soma bin/somasc.sombin

runtime:
	python runtime/soma_runtime.py bin/examples.sombin

clean:
	rm -f bin/*.sombin
