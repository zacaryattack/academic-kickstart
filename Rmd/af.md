#  Example of How to Plot Activation Functions

This example shows how to plot the activation function of a neuron with one input.

Farhad kamangar 2018


```python
!pip3 install ipywidgets


from ipywidgets import interact, interactive, fixed, interact_manual

```

    Collecting ipywidgets
      Downloading ipywidgets-7.5.1-py2.py3-none-any.whl (121 kB)
    [K     |████████████████████████████████| 121 kB 1.4 MB/s eta 0:00:01
    [?25hRequirement already satisfied: traitlets>=4.3.1 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from ipywidgets) (4.3.3)
    Requirement already satisfied: ipykernel>=4.5.1 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from ipywidgets) (5.1.3)
    Collecting widgetsnbextension~=3.5.0
      Downloading widgetsnbextension-3.5.1-py2.py3-none-any.whl (2.2 MB)
    [K     |████████████████████████████████| 2.2 MB 21 kB/s eta 0:00:013
    [?25hRequirement already satisfied: ipython>=4.0.0; python_version >= "3.3" in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from ipywidgets) (7.11.1)
    Requirement already satisfied: nbformat>=4.2.0 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from ipywidgets) (5.0.4)
    Requirement already satisfied: ipython-genutils in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from traitlets>=4.3.1->ipywidgets) (0.2.0)
    Requirement already satisfied: six in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from traitlets>=4.3.1->ipywidgets) (1.12.0)
    Requirement already satisfied: decorator in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from traitlets>=4.3.1->ipywidgets) (4.4.1)
    Requirement already satisfied: appnope; platform_system == "Darwin" in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from ipykernel>=4.5.1->ipywidgets) (0.1.0)
    Requirement already satisfied: jupyter-client in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from ipykernel>=4.5.1->ipywidgets) (5.3.4)
    Requirement already satisfied: tornado>=4.2 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from ipykernel>=4.5.1->ipywidgets) (6.0.3)
    Requirement already satisfied: notebook>=4.4.1 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from widgetsnbextension~=3.5.0->ipywidgets) (6.0.3)
    Requirement already satisfied: setuptools>=18.5 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets) (41.0.1)
    Requirement already satisfied: pexpect; sys_platform != "win32" in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets) (4.8.0)
    Requirement already satisfied: backcall in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets) (0.1.0)
    Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets) (3.0.2)
    Requirement already satisfied: jedi>=0.10 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets) (0.15.2)
    Requirement already satisfied: pickleshare in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets) (0.7.5)
    Requirement already satisfied: pygments in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets) (2.3.1)
    Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from nbformat>=4.2.0->ipywidgets) (3.2.0)
    Requirement already satisfied: jupyter-core in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from nbformat>=4.2.0->ipywidgets) (4.6.1)
    Requirement already satisfied: pyzmq>=13 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from jupyter-client->ipykernel>=4.5.1->ipywidgets) (18.1.1)
    Requirement already satisfied: python-dateutil>=2.1 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from jupyter-client->ipykernel>=4.5.1->ipywidgets) (2.8.1)
    Requirement already satisfied: nbconvert in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (5.6.1)
    Requirement already satisfied: terminado>=0.8.1 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.8.3)
    Requirement already satisfied: prometheus-client in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.7.1)
    Requirement already satisfied: Send2Trash in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (1.5.0)
    Requirement already satisfied: jinja2 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (2.10.1)
    Requirement already satisfied: ptyprocess>=0.5 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from pexpect; sys_platform != "win32"->ipython>=4.0.0; python_version >= "3.3"->ipywidgets) (0.6.0)
    Requirement already satisfied: wcwidth in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->ipython>=4.0.0; python_version >= "3.3"->ipywidgets) (0.1.8)
    Requirement already satisfied: parso>=0.5.2 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from jedi>=0.10->ipython>=4.0.0; python_version >= "3.3"->ipywidgets) (0.5.2)
    Requirement already satisfied: attrs>=17.4.0 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets) (19.3.0)
    Requirement already satisfied: pyrsistent>=0.14.0 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets) (0.15.7)
    Requirement already satisfied: importlib-metadata; python_version < "3.8" in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets) (0.23)
    Requirement already satisfied: entrypoints>=0.2.2 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.3)
    Requirement already satisfied: testpath in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.4.4)
    Requirement already satisfied: pandocfilters>=1.4.1 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (1.4.2)
    Requirement already satisfied: defusedxml in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.6.0)
    Requirement already satisfied: bleach in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (3.1.0)
    Requirement already satisfied: mistune<2,>=0.8.1 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.8.4)
    Requirement already satisfied: MarkupSafe>=0.23 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from jinja2->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (1.1.1)
    Requirement already satisfied: zipp>=0.5 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets) (0.6.0)
    Requirement already satisfied: webencodings in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from bleach->nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.5.1)
    Requirement already satisfied: more-itertools in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets) (7.2.0)
    Installing collected packages: widgetsnbextension, ipywidgets
    Successfully installed ipywidgets-7.5.1 widgetsnbextension-3.5.1



```python
%matplotlib inline
from ipywidgets import interactive
import matplotlib.pyplot as plt
import numpy as np

def plot_activation(weight, bias,activation_type):
    plt.figure(2)
    input_array = np.linspace(-10, 10, num=100)
    net = weight * input_array + bias
   
    if activation_type=="Linear":
        activation = net
    elif activation_type=="Hard Limit":      
        net[net>=0]=1
        net[net<0]=0
        activation=net
    elif activation_type=="Sigmoid":
        activation = 1.0 / (1 + np.exp(-net))
    elif activation_type=="Relu":
        net[net<0]=0
        activation=net
    plt.plot(input_array,activation)
    plt.ylim(-10, 10)
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid(True)
    plt.show()

interactive_plot = interactive(plot_activation,
                               weight=widgets.FloatSlider(min=-10,max=10,step=1,value=1),
                               bias=widgets.FloatSlider(min=-10,max=10,step=1,value=0),
                              activation_type=widgets.Dropdown(options=['Linear','Hard Limit','Sigmoid', 'Relu'],
    value='Linear',
    description='Activation Type:',
    disabled=False,
))
interactive_plot
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-5-fb72dc96a8a9> in <module>
    ----> 1 get_ipython().run_line_magic('matplotlib', 'inline')
          2 from ipywidgets import interactive
          3 import matplotlib.pyplot as plt
          4 import numpy as np
          5 


    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/IPython/core/interactiveshell.py in run_line_magic(self, magic_name, line, _stack_depth)
       2305                 kwargs['local_ns'] = sys._getframe(stack_depth).f_locals
       2306             with self.builtin_trap:
    -> 2307                 result = fn(*args, **kwargs)
       2308             return result
       2309 


    </Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/decorator.py:decorator-gen-108> in matplotlib(self, line)


    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/IPython/core/magic.py in <lambda>(f, *a, **k)
        185     # but it's overkill for just that one bit of state.
        186     def magic_deco(arg):
    --> 187         call = lambda f, *a, **k: f(*a, **k)
        188 
        189         if callable(arg):


    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/IPython/core/magics/pylab.py in matplotlib(self, line)
         97             print("Available matplotlib backends: %s" % backends_list)
         98         else:
    ---> 99             gui, backend = self.shell.enable_matplotlib(args.gui.lower() if isinstance(args.gui, str) else args.gui)
        100             self._show_matplotlib_backend(args.gui, backend)
        101 


    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/IPython/core/interactiveshell.py in enable_matplotlib(self, gui)
       3393         """
       3394         from IPython.core import pylabtools as pt
    -> 3395         gui, backend = pt.find_gui_and_backend(gui, self.pylab_gui_select)
       3396 
       3397         if gui != 'inline':


    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/IPython/core/pylabtools.py in find_gui_and_backend(gui, gui_select)
        274     """
        275 
    --> 276     import matplotlib
        277 
        278     if gui and gui != 'auto':


    ModuleNotFoundError: No module named 'matplotlib'



```python

```
