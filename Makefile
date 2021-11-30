# definisce la macro CPPC
ifndef CPPC
	CPPC=g++
endif

INPUT_CONFIG="./data/2006/2006_000000000000.cfg"
OUTPUT_CONFIG="./data/2006_OUT/output_2006"
STEPS=100

# definisce le macro contenenti i nomei degli eseguibili
# e il numero di thread omp per la versione parallela
NT = 2 # numero di threads OpenMP
EXEC = sciara_omp
EXEC_SERIAL = sciara_serial

# definisce il target di default, utile in
# caso di invocazione di make senza parametri
default:all

# compila le versioni seriale e OpenMP
all:
	$(CPPC) *.cpp -o $(EXEC_SERIAL) -g
#	$(CPPC) *.cpp -o $(EXEC) -fopenmp -O3


# esegue la simulazione OpenMP
run_omp:
	OMP_NUM_THREADS=$(NT) ./$(EXEC) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) 
# &&  md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)

# esegue la simulazione seriale 
run:
	./$(EXEC_SERIAL) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) 
# &&  md5sum $(OUT_SERIAL) && cat $(HDR) $(OUT_SERIAL) > $(OUT_SERIAL).qgis && rm $(OUT_SERIAL)

# elimina l'eseguibile, file oggetto e file di output
clean:
	rm -f $(EXEC) $(EXEC_SERIAL) *.o *output*

# elimina file oggetto e di output
wipe:
	rm -f *.o *output*
