import numpy


def bytes(length):
    """Returns random bytes.

    .. seealso:: :func:`numpy.random.bytes`
    """
    return numpy.bytes(length)


from clpy.random import distributions  # NOQA
from clpy.random import generator  # NOQA
from clpy.random import permutations  # NOQA
from clpy.random import sample as sample_  # NOQA


# import class and function
from clpy.random.distributions import gumbel  # NOQA
from clpy.random.distributions import lognormal  # NOQA
from clpy.random.distributions import normal  # NOQA
from clpy.random.distributions import standard_normal  # NOQA
from clpy.random.distributions import uniform  # NOQA
from clpy.random.generator import get_random_state  # NOQA
from clpy.random.generator import RandomState  # NOQA
from clpy.random.generator import reset_states  # NOQA
from clpy.random.generator import seed  # NOQA
from clpy.random.permutations import shuffle  # NOQA
from clpy.random.sample import choice  # NOQA
from clpy.random.sample import multinomial  # NOQA
from clpy.random.sample import rand  # NOQA
from clpy.random.sample import randint  # NOQA
from clpy.random.sample import randn  # NOQA
from clpy.random.sample import random_integers  # NOQA
from clpy.random.sample import random_sample  # NOQA
from clpy.random.sample import random_sample as random  # NOQA
from clpy.random.sample import random_sample as ranf  # NOQA
from clpy.random.sample import random_sample as sample  # NOQA
