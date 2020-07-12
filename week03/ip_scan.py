import sys
from subprocess import Popen, PIPE
from concurrent.futures import ThreadPoolExecutor, as_completed
import socket
import json

def ping(ip):
    p = Popen(['ping', '-c', '1', ip], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    is_success = 'ttl' in str(output)
    return is_success, ip

def parse_ip_range(ip_range):
    ip_start, ip_end = ip_range.split('-')
    ip_start_subs = ip_start.split('.')
    ip_end_subs = ip_end.split('.')
    ip_list = []

    for i in range(int(ip_start_subs[3]), int(ip_end_subs[3])+1):
        if i < 256:
            ip_list.append(f'{ip_start_subs[0]}.{ip_start_subs[1]}.{ip_start_subs[2]}.{str(i)}')

    return ip_list

def do_ping(ip_list, thread_count):
    with ThreadPoolExecutor(thread_count) as executor:
        print('output')
        for r in executor.map(ping, parse_ip_range(ip_list)):
            if r[0]:
                print(r[1])

def try_tcp(s, ip, port):
    result = s.connect_ex((ip, port))
    return result == 0, port

def port_scan_list(s, ip):
    return [(s, ip, i) for i in range(1, 1025)]

def do_tcp(ip, thread_count, output_file):
    results = []
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with ThreadPoolExecutor(thread_count) as executor:
        for r in executor.map(lambda args:try_tcp(*args), port_scan_list(s, ip)):
            if r[0]:
                results.append(r[1])
    
    with open(output_file, 'w') as o:
        json.dump(results, o)


thread_count = 1
if sys.argv[1] == '-n':
    thread_count = int(sys.argv[2])

ip = '127.0.0.1'
if sys.argv[5] == '-ip':
    ip = sys.argv[6]


print(f'Thread count: {thread_count}')
print(f'IP: {ip}')

if sys.argv[3] == '-f':
    if sys.argv[4] == 'ping':
        do_ping(ip, thread_count)
    else:
        output_file = './result.json'
        if sys.argv[7] == '-w':
            output_file = sys.argv[8]

        print(f'Output: {output_file}')
        do_tcp(ip, thread_count, output_file)

