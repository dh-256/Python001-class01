学习笔记

## Scrapy 并发参数优化原理
	* scrapy 使用了 requests，效率更高
	* requests 是同步请求数据
	* scrapy 是异步请求数据

setting.py
默认最大并发连接数，根据情况不要太高 默认 16
CONCURRENT_REQUESTS=32

默认一个网站的延迟，防止爬取过快 默认 0
DOWNLOAD_DELAY=3

针对域名和 ip 的并发量限制，两个当中只会 honor 一个
CONCURRENT_REQUESTS_PER_DOMAIN=16
CONCURRENT_REQUESTS_PER_IP=16

## 多进程
多进程模型
多进程，多线程，协程的目的都是希望尽可能多处理任务
产生新的进程可以使用以下方式
os.fork()    // only supports linux / mac
multiprocessing.Process()

进程的父子关系
fork 以后父进程和子进程都会执行后面的代码
res = os.fork()，res == 0 的话是子进程，否则是父进程
os.getpid() 是 pid

多进程程序调试技巧
os.getpid() // pid
os.getppid() // 父进程 pid
for p in multiprocessing.active_children() // 当前父进程下活动的子进程
multiprocessing.cpu_count()  // 获取计算机 cpu 核心个数

通过继承的方式创建新的进程
from multiprocessing import Process
class NewProcess(Process)
    def __init__
    def __run(self):

使用队列实现进程间通信
主要共享方式
队列 multiprocessing.Queue
管道
共享内存

资源的抢占
加锁机制

multiprocessing.Queue
建议设置最大 queue size

管道
multiprocessing.Pipe

共享内存
multiprocessing.Value
multiprocessing.Array

进程池
multiprocessing.pool.Pool
p = Pool(4)
p.apply_async(f, args=(...)) // 增加异步任务
p.apply(f, args=(...)) // 增加同步任务
p.close() // 关闭池子，防止增加新的任务，会等待任务结束
p.join() // 等待子进程结束，前面一定要有 close 或者 terminated，不然父进程可能会一直等
p.terminate() // 强制结束

多线程
进程和线程的区别
多个线程在一个进程之间
线程之间共享内存空间

线程的同步/阻塞，异步/非阻塞
阻塞/非阻塞是调用方看到的结果，发起之后还能不能做别的事情
同步/异步是非调用方看到的结果，马上响应还是等一会响应

为什么有多进程还要有多线程（python 特点）
多线程只能在一个 CPU/物理核心 上运行
多进程可以在多个 CPU/物理核心上运行

为什么用协程
进程/线程的调度是系统控制
协程的调度是开发者控制

并发和并行
并发 2 queues 1 coffee machine
并行 2 queues 2 coffee machines

threading 模块
使用函数创建
theading.Thread(target=run, args=("thread 1", ))
使用类创建
class MyThread(threading.Thread)

线程锁
mutex = threading.Lock()    // 普通锁，不可嵌套
mutex.acquire()
mutex.release()

threading.RLock()    // reentrant lock，可以嵌套
条件锁
conn = threading.Condition()
conn.acquire()
conn.wait_for(condition)
conn.release()

信号量
内部放一个计数器，占用信号量的线程超过一个数字就阻塞
semaphore = threading.BoundedSemaphore(5)
semaphore.acquire()
semaphore.release()

事件
event = threading.Event()
event.set()
event.clear()
e.wait()

定时器
from theading import Timer
t = Timer(1, f)

线程池
一般的线程池
from multiprocessing.dummy import Pool as ThreadPool

并行任务的高级封装(3.2 以后)
from concurrent.futures import ThreadPoolExecutor

一般线程池
pool = ThreadPool(5)
results = pool.map(f, list)
pool.close()
pool.join()

GIL 锁
python 解释器有区别
最常见是 cpython，有 GIL 全局锁，多线程是伪多线程
GIL 全局解释锁 global interpreter lock
每个进程只有一个 GIL 锁
拿到 GIL 锁可以使用 CPU
cpython 解释器不是真正意思上的多线程，属于伪并发