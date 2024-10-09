+++
title = "HTTP-C"
description = "A POSIX HTTP server written in C"
weight = 1

date="2024-10-05"
[taxonomies]
tags=["C", "projects"]
[extra]
# You can also crop the image in the url by adjusting w=/h=
remote_image = "https://mdpan.dev/images/http-project-image.png"
+++

<!-- link_to = "https://github.com/mariodanielpanuco/http-c" -->

## Usage

```
./httpserver [-t threads_count] [-l log_path] <port> 
```

## Methods
Implemented GET, PUT, and APPEND 


### Performance 
TODO: Fill out from report
<!-- ## Performance with 4 threads -->
<!-- - GET: 5 gb/s  -->
<!-- - PUT:  -->
<!-- - APPEND:  -->
<!---->


### POSIX 
Use just System(1) and System(2) functions, i.e., no calls to System(3) and above.

## Libraries/Modules 
### opt.h
- enum OPTS
- opt_parse()
- struct opt {
    n_threads: number of worker threads 
    log_file: pathname for the log file
}

### dictionary.h
Used to manage the HTTP request header fields
- dict_init()
- hash()
- dict_get(key) -> value
- dict_put(key, value): Puts the value in a hash bucket assigned to it by the key string

### list.h
- list_init(ssize_t size): Inializes a list with a max capacity of _size_

### queue.h
- queue_new(int size) -> Queue_t*: Initializes a queue, with capacity _size_, for storing functions and arguments.
- queue_delete(int size): 
- queue_push((void*) func, (void*)args): pushes a processes (function and it's arguments) to the queue
- queue_pop(Queue_t *q): Returns the function and 
- queue_print(Queue_t *q): Prints the queue

### listener.h
- listener_new(ssize_t connfd): Creates a listener struct for the designated port's file descriptor connfd
- listener_accept(Listener lst): Listener accepts a connection from the connfd

### request.h
- parse_request()
- request_get_method()

### response.h
- Enum Response_Code {
    SUCCESS, 
    INTERNAL_SERVER_ERROR,
    FILE_NOT_FOUND,
    NEW_FILE, 
    BAD_REQUEST,
    FORBIDDEN,
    NOT_IMPLEMENTED,
    VERSION_NOT_SUPPORTED
}
- struct Response {
    enum Response_code;
    Request rqst;
}
- response_get_code(struct Response resp): Returns the appropriate response to be sent to the client.
- response_get_method(struct Response resp): Returns the request method (GET, PUT, APPEND)


### util.h
- readBytez(ssize_t N, usize_t connfd): Reads N bytes from the file descriptor connfd
- writeBytez(usize_t connfd, char * msg): Writes all the bytes in msg to the file descriptor connfd
- passBytez(ssize_t n): Skip the next N bytes 
- sig_term: Handles termate signals from the command line
- sig_int: Handles interupt signals from the command line
- audit_log(): Writes a string to the log file
- Other misc. functions

