# Crtanje grafika: poređenje predviđenih i stvarnih cijena na test skupu.

import matplotlib.pyplot as plt


class CustomPlotting:

    def show_plots(self, testPredict, testY):
        """Crta dvije linije: predikciju i stvarne vrijednosti, po jednu tačku za svaku kuću iz test skupa."""
        plt.plot(testPredict, label='Predikcija')
        plt.plot(testY, label='Stvarna cijena')
        plt.xlabel('Redni broj kuće u test skupu')
        plt.ylabel('Cijena (hiljade $)')
        plt.legend()
        plt.show()        # blokira izvršavanje dok se prozor sa grafikom ne zatvori
