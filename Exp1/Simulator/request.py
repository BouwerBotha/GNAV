from threading import Thread
import subprocess
import socket
import time
import numpy as np
import tensorflow as tf
import math
from common import NUM_DIM,WIDTH,HEIGHT,NUM_SIMS
HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 11001  # The port used by the server
print('instantiating %d servers' %NUM_SIMS)




processes = []
servers = []



for t in range(NUM_SIMS):
    port = PORT + t
    process = subprocess.Popen(['../Game/objects/My Project.exe',str(port)])
    # process = subprocess.Popen(['../Game/{0}/My Project.exe'.format(3), str(port)])
    processes.append(process)

time.sleep(10)

for t in range(NUM_SIMS):
    server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    port = PORT + t
    server.connect(("127.0.0.1", port))
    print('Connected on: %d' % port)
    servers.append(server)


def end():
    print('Terminating servers and sockets')
    for s in servers:
        s.close()
    for p in processes:
        p.kill()


def request_images(target_loc,avoid_loc, x,y,bearing,num_threads = NUM_SIMS):
    ##shift all coords forward by 10.5 cm
    local_x_change = tf.cast(tf.zeros_like(x),dtype=tf.float32)
    local_y_change = tf.cast(tf.ones_like(y)*0.105,dtype=tf.float32)

    local_dist_change = tf.sqrt(tf.square(local_x_change) + tf.square(local_y_change))
    bearing_change = tf.cast((-tf.math.atan2(local_y_change, local_x_change) + (math.pi / 2)) % (math.pi * 2.0),dtype=tf.float32)

    temp_bearing_c = tf.cast(bearing / 180.0 * math.pi,dtype=tf.float32) + bearing_change
    global_x_change = local_dist_change * tf.sin(temp_bearing_c)
    global_y_change = local_dist_change * tf.cos(temp_bearing_c)

    x = tf.cast(x,tf.float32)+global_x_change
    y = tf.cast(y,tf.float32)+global_y_change
    if str(type(x)) == "<class 'tensorflow.python.framework.ops.EagerTensor'>":
        x = x.numpy()
        y = y.numpy()

    if str(type(bearing)) == "<class 'tensorflow.python.framework.ops.EagerTensor'>":
        bearing = bearing.numpy()

    target_loc = target_loc.astype(np.float32)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    bearing = bearing.astype(np.float32)
    avoid_loc = avoid_loc.astype(np.float32)
    NUM_IMAGES = x.shape[1]
    imgs = np.zeros([num_threads, NUM_IMAGES, HEIGHT,WIDTH, NUM_DIM], dtype=np.uint8)

    def one_thread(target_pos,avoid_pos,x,y,bearing,server_id):
        big_array = np.zeros([NUM_IMAGES, HEIGHT, WIDTH, NUM_DIM])
        server = servers[server_id]

        string_to_send = str(target_pos[0])
        string_to_send2 = str(target_pos[1])
        string_to_send3 = str(avoid_pos[0])
        string_to_send4 = str(avoid_pos[1])
        width = str(WIDTH).encode()
        height = str(HEIGHT).encode()
        str_image_num = str(NUM_IMAGES).encode()
        server.sendall(str_image_num + b"," +width+b"," +height+b"," + bytearray(string_to_send.encode()) + b"," + bytearray(
        string_to_send2.encode()) + b"," + bytearray(string_to_send3.encode()) + b"," + bytearray(
        string_to_send4.encode()) + b",%" + x.tobytes() + y.tobytes() + bearing.tobytes())

        for i in range(NUM_IMAGES):
            num_bytes = 0
            data = server.recv(HEIGHT * WIDTH * NUM_DIM)
            num_bytes += len(data)

            while (num_bytes < HEIGHT * WIDTH * NUM_DIM):
                second = server.recv(HEIGHT * WIDTH * NUM_DIM - num_bytes)
                data = data + second
                num_bytes += len(second)

            big_array[i] = np.reshape(np.frombuffer(data, np.uint8), [HEIGHT, WIDTH, 3])[::-1, :, :]

        imgs[server_id]=big_array


    threads = []
    for i in range(num_threads):
        thread = Thread(target=one_thread, args=(target_loc[i],avoid_loc[i],x[i],y[i],bearing[i],i))
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()



    return imgs

