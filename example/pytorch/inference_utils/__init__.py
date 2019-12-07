#
# Modified from source
# Source: https://stackoverflow.com/questions/17667903/python-socket-receive-large-amount-of-data
# othterwise it is hard to send / recv potentially large amount of data 
#
import struct
import pickle

# use this to send
# use pickle will use pickle to pack the msg
def send_msg(sock, msg, use_pickle=False):
    # Prefix each message with a 4-byte length (network byte order)
    if use_pickle:
        msg = pickle.dumps(msg)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

# helper method for receive
# will block untill all data have been received
def _recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

# use this to receive
# use pickle will use pickle to try to unpack the msg
def recv_msg(sock, use_pickle=False):
    # Read message length and unpack it into an integer
    raw_msglen = _recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    recv_data = _recvall(sock, msglen)

    if use_pickle:
        recv_data = pickle.loads(recv_data)

    return recv_data 
