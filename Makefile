GRAY = NoisyGrayStep.png CleanGrayStep.png
EGRAY = egray/egray.s.0.030000.n.0.300000.d.0.300000.gif

all:	$(GRAY) $(EGRAY)

.PHONY: clean
clean:
	rm -fr build gray egray NoisyGrayStep.png CleanGrayStep.png

gray:	Makefile
	mkdir -p gray
	./gray.py

egray:	Makefile
	mkdir -p egray
	./egray.py
	./egray.py --step=1e-1 --noise=3e-1

build:	Makefile
	mkdir -p build

$(EGRAY): egray Makefile

$(GRAY): gray Makefile
	convert gray/gray.step.0.030000.scale.0.300000.gif NoisyGrayStep.png
	convert gray/gray.step.0.030000.scale.0.000000.gif CleanGrayStep.png
