bioinf_code folder:
    contiene i risultati della rete siamese allenata e testata sul mio dataset (sia su 4 che su 5 classi)


paper_code folder:
    - train_test folder
        contiene i risultati della rete siamese allenata e testata sia sul mio dataset (4 - 5 classi) che su TON_IoT (4 classi)

    - test folder
        - TON_IoT / only_test folder
            contiene i risultati della rete siamese pre-addestrata sul mio dataset (4 - 5 claasi) e testata su TON_IoT

        - TON_IoT / transfer_learning folder
            contiene i risultati della rete siamese pre-addestrata sul mio dataset (4 - 5 claasi) e aggiornata
            con il dataset TON_IoT usando il transfer learning (lr = 0.000001, epochs = 40, patience = 5)