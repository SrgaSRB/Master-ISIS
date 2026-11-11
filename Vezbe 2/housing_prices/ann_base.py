# Bazna klasa za neuronsku mrežu (ANN = Artificial Neural Network).
# Ovdje se čuvaju SVI hiperparametri mreže (broj epoha, batch, optimizator, ...),
# a konkretna arhitektura modela se gradi u podklasi (ann_regression.py).


# ----------------------------- Podrazumijevani hiperparametri -----------------------------
EPOCH_NUMBER = 100                 # broj epoha = koliko puta mreža "prođe" kroz cijeli trening skup
BATCH_SIZE_NUMBER = 1              # veličina batch-a = poslije koliko uzoraka se ažuriraju težine
                                   # (1 = "online" učenje; najsporije, ali najčešće ažuriranje težina)
COST_FUNCTION = 'mean_squared_error'  # funkcija greške (loss) koju mreža minimizuje - MSE je standard za regresiju
OPTIMIZER = 'adam'                 # algoritam za ažuriranje težina (Adam = adaptivna stopa učenja, dobar default)
KERNEL_INITIALIZER = 'normal'      # kako se inicijalizuju početne težine (normalna/Gausova raspodjela)
ACTIVATION_FUNCTION = 'sigmoid'    # aktivaciona funkcija skrivenih slojeva (izlaz u opsegu 0..1)
HIDDEN_LAYERS = (20, 8)            # broj neurona po skrivenom sloju
VERBOSE = 2                        # koliko Keras ispisuje tokom treniranja (0 = ništa, 1 = progress bar, 2 = jedna linija po epohi)


class AnnBase:

    def __init__(
        self,
        epoch_number=EPOCH_NUMBER,
        batch_size_number=BATCH_SIZE_NUMBER,
        cost_function=COST_FUNCTION,
        optimizer=OPTIMIZER,
        kernel_initializer=KERNEL_INITIALIZER,
        activation_function=ACTIVATION_FUNCTION,
        hidden_layers=HIDDEN_LAYERS,
        verbose=VERBOSE,
        l2_strength=0.0,
        dropout_rate=0.0,
    ):
        self.epoch_number = epoch_number
        self.batch_size_number = batch_size_number
        self.activation_function = activation_function
        self.cost_function = cost_function
        self.optimizer = optimizer
        self.kernel_initializer = kernel_initializer
        self.hidden_layers = tuple(hidden_layers)
        self.verbose = verbose
        self.l2_strength = l2_strength
        self.dropout_rate = dropout_rate

    # ------------------------------------------------------------------------------------
    # Ispod su getter/setter parovi (Python @property) za svaki hiperparametar.
    # Svaki samo čita/piše "privatni" atribut sa donjom crtom (npr. self._epoch_number).
    # Funkcionalno su ekvivalentni običnim atributima; ovdje služe kao mjesto gdje bi se
    # kasnije mogla dodati validacija (npr. da broj epoha mora biti > 0).
    # ------------------------------------------------------------------------------------

    @property
    def epoch_number(self):
        return self._epoch_number

    @epoch_number.setter
    def epoch_number(self, value):
        self._epoch_number = value

    @property
    def batch_size_number(self):
        return self._batch_size_number

    @batch_size_number.setter
    def batch_size_number(self, value):
        self._batch_size_number = value

    # Funkcija greške (loss) - veličina koju model pokušava da minimizuje tokom treniranja.
    # Dozvoljene vrijednosti u Kerasu (za regresiju):
    #mean_squared_error
    #mean_absolute_error
    #mean_absolute_percentage_error
    #mean_squared_logarithmic_error
    #cosine_similarity
    #huber_loss
    @property
    def cost_function(self):
        return self._cost_function

    @cost_function.setter
    def cost_function(self, value):
        self._cost_function = value

    # Optimizator - algoritam koji na osnovu gradijenta funkcije greške ažurira težine.
    # Dozvoljene vrijednosti u Kerasu:
    #SGD
    #RMSprop
    #adam
    #adadelta
    #adagrad
    #adamax
    #nadam
    #ftrl
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    # Inicijalizator - definiše kako se postavljaju početne (slučajne) težine slojeva.
    # Dozvoljene vrijednosti u Kerasu:
    #random_normal
    #normal
    #random_uniform
    #zeros
    #ones
    #glorot_normal
    #glorot_uniform
    #he_normal
    #he_uniform
    #identity
    @property
    def kernel_initializer(self):
        return self._kernel_initializer

    @kernel_initializer.setter
    def kernel_initializer(self, value):
        self._kernel_initializer = value

    # Aktivaciona funkcija - nelinearnost koja se primjenjuje na izlaz svakog neurona.
    # Dozvoljene vrijednosti u Kerasu:
    #relu
    #sigmoid
    #softmax
    #softplus
    #softsign
    #tanh
    #selu
    #elu
    #exponential
    @property
    def activation_function(self):
        return self._activation_function

    @activation_function.setter
    def activation_function(self, value):
        self._activation_function = value

    # Arhitektura skrivenih slojeva: broj neurona u svakom sloju.
    @property
    def hidden_layers(self):
        return self._hidden_layers

    @hidden_layers.setter
    def hidden_layers(self, value):
        self._hidden_layers = value

    # Nivo ispisa tokom treniranja (0, 1 ili 2).
    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value
