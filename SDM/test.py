
from Model.Frame import Frame
from Model.Sequence import ZadoffSequence, DFTSequence
from Model.utils import create_cor_matirix

user = 19
pilot_len = 9
size = 19
path = 1

pilot = Frame(user, pilot_len).create_frame_matrix()

dft_pilot = DFTSequence(user, path, pilot_len, size).create_pilot()

create_cor_matirix(pilot)
print('---------------------------------------------------------------------------')
create_cor_matirix(dft_pilot)