IMG=evcomx-rf:latest
build:    ; docker build -t $(IMG) .
run:      ; docker run --rm -p 8501:8501 $(IMG)
verify:   ; docker run --rm $(IMG) python pipeline.py verify
evaluate: ; docker run --rm -v $(PWD)/data_out:/app/data_out --user root $(IMG) python pipeline.py evaluate --output data_out/avaliacao.json
predict:  ; docker run --rm --user root -v $(PWD)/data_in:/app/data_in:ro -v $(PWD)/data_out:/app/data_out $(IMG) python pipeline.py predict --cestao $(CESTAO) --input data_in/entrada.csv --output data_out/previsoes_c$(CESTAO).csv
