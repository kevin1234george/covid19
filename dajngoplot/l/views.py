from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
from matplotlib import pylab
from pylab import *
import PIL, PIL.Image
from io import BytesIO
import matplotlib
import pandas as pd
import seaborn as sb

matplotlib.use('Agg')
from matplotlib import pyplot

df = pd.read_csv('/home/astra/Desktop/sea/dajngoplot/l/data.csv')


def index(request):
    lookup = request.GET.get('country')
    global x
    x = lookup
    c = {'data': list(set(df.Country.values))}

    return render(request, 'index.html', c)


def showimage(request):
    df = pd.read_csv('/home/astra/Desktop/sea/dajngoplot/l/data.csv')
    sb.set_style("ticks")
    if x is not None:
        df = df[df.Country == x]
    if x == 'nan':
        df = pd.read_csv('/home/astra/Desktop/sea/dajngoplot/l/data.csv')
    sb.lmplot(data=df, x="Cumulative Deaths", y="Cumulative Confirmed", hue="Region", fit_reg=False)

    buffer = BytesIO()
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer, "PNG")
    pylab.close()

    return HttpResponse(buffer.getvalue(), content_type="image/png")




def alldata(request):
    sb.set_style("ticks")
    sb.pairplot(df, hue='Region', diag_kind="kde", kind="scatter", palette="husl")

    buffer = BytesIO()
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer, "PNG")
    pylab.close()

    return HttpResponse(buffer.getvalue(), content_type="image/png")
