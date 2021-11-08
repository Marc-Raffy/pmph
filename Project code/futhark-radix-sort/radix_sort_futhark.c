#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wparentheses"
#pragma GCC diagnostic ignored "-Wunused-label"
#endif
#ifdef __clang__
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wparentheses"
#pragma clang diagnostic ignored "-Wunused-label"
#endif
// Headers

#define _GNU_SOURCE
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <float.h>
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#define CL_SILENCE_DEPRECATION
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif


// Initialisation

int futhark_get_num_sizes(void);
const char *futhark_get_size_name(int);
const char *futhark_get_size_class(int);
struct futhark_context_config ;
struct futhark_context_config *futhark_context_config_new(void);
void futhark_context_config_free(struct futhark_context_config *cfg);
void futhark_context_config_add_build_option(struct futhark_context_config *cfg,
                                             const char *opt);
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag);
void futhark_context_config_set_profiling(struct futhark_context_config *cfg,
                                          int flag);
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag);
void futhark_context_config_set_device(struct futhark_context_config *cfg, const
                                       char *s);
void futhark_context_config_set_platform(struct futhark_context_config *cfg,
                                         const char *s);
void
futhark_context_config_select_device_interactively(struct futhark_context_config *cfg);
void futhark_context_config_list_devices(struct futhark_context_config *cfg);
void futhark_context_config_dump_program_to(struct futhark_context_config *cfg,
                                            const char *path);
void
futhark_context_config_load_program_from(struct futhark_context_config *cfg,
                                         const char *path);
void futhark_context_config_dump_binary_to(struct futhark_context_config *cfg,
                                           const char *path);
void futhark_context_config_load_binary_from(struct futhark_context_config *cfg,
                                             const char *path);
void
futhark_context_config_set_default_group_size(struct futhark_context_config *cfg,
                                              int size);
void
futhark_context_config_set_default_num_groups(struct futhark_context_config *cfg,
                                              int num);
void
futhark_context_config_set_default_tile_size(struct futhark_context_config *cfg,
                                             int num);
void
futhark_context_config_set_default_threshold(struct futhark_context_config *cfg,
                                             int num);
int futhark_context_config_set_size(struct futhark_context_config *cfg, const
                                    char *size_name, size_t size_value);
struct futhark_context ;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg);
struct futhark_context
*futhark_context_new_with_command_queue(struct futhark_context_config *cfg,
                                        cl_command_queue queue);
void futhark_context_free(struct futhark_context *ctx);
cl_command_queue futhark_context_get_command_queue(struct futhark_context *ctx);

// Arrays

struct futhark_i32_1d ;
struct futhark_i32_1d *futhark_new_i32_1d(struct futhark_context *ctx, const
                                          int32_t *data, int64_t dim0);
struct futhark_i32_1d *futhark_new_raw_i32_1d(struct futhark_context *ctx, const
                                              cl_mem data, int offset,
                                              int64_t dim0);
int futhark_free_i32_1d(struct futhark_context *ctx,
                        struct futhark_i32_1d *arr);
int futhark_values_i32_1d(struct futhark_context *ctx,
                          struct futhark_i32_1d *arr, int32_t *data);
cl_mem futhark_values_raw_i32_1d(struct futhark_context *ctx,
                                 struct futhark_i32_1d *arr);
const int64_t *futhark_shape_i32_1d(struct futhark_context *ctx,
                                    struct futhark_i32_1d *arr);

// Opaque values


// Entry points

int futhark_entry_main(struct futhark_context *ctx,
                       struct futhark_i32_1d **out0, const
                       struct futhark_i32_1d *in0);

// Miscellaneous

int futhark_context_sync(struct futhark_context *ctx);
char *futhark_context_report(struct futhark_context *ctx);
char *futhark_context_get_error(struct futhark_context *ctx);
void futhark_context_set_logging_file(struct futhark_context *ctx, FILE *f);
void futhark_context_pause_profiling(struct futhark_context *ctx);
void futhark_context_unpause_profiling(struct futhark_context *ctx);
int futhark_context_clear_caches(struct futhark_context *ctx);
#define FUTHARK_BACKEND_opencl
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>
#undef NDEBUG
#include <assert.h>
#include <stdarg.h>
// Start of util.h.
//
// Various helper functions that are useful in all generated C code.

#include <errno.h>
#include <string.h>

static const char *fut_progname = "(embedded Futhark)";

static void futhark_panic(int eval, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "%s: ", fut_progname);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  exit(eval);
}

// For generating arbitrary-sized error messages.  It is the callers
// responsibility to free the buffer at some point.
static char* msgprintf(const char *s, ...) {
  va_list vl;
  va_start(vl, s);
  size_t needed = 1 + (size_t)vsnprintf(NULL, 0, s, vl);
  char *buffer = (char*) malloc(needed);
  va_start(vl, s); // Must re-init.
  vsnprintf(buffer, needed, s, vl);
  return buffer;
}


static inline void check_err(int errval, int sets_errno, const char *fun, int line,
                            const char *msg, ...) {
  if (errval) {
    char errnum[10];

    va_list vl;
    va_start(vl, msg);

    fprintf(stderr, "ERROR: ");
    vfprintf(stderr, msg, vl);
    fprintf(stderr, " in %s() at line %d with error code %s\n",
            fun, line,
            sets_errno ? strerror(errno) : errnum);
    exit(errval);
  }
}

#define CHECK_ERR(err, msg...) check_err(err, 0, __func__, __LINE__, msg)
#define CHECK_ERRNO(err, msg...) check_err(err, 1, __func__, __LINE__, msg)

// Read a file into a NUL-terminated string; returns NULL on error.
static void* slurp_file(const char *filename, size_t *size) {
  unsigned char *s;
  FILE *f = fopen(filename, "rb"); // To avoid Windows messing with linebreaks.
  if (f == NULL) return NULL;
  fseek(f, 0, SEEK_END);
  size_t src_size = ftell(f);
  fseek(f, 0, SEEK_SET);
  s = (unsigned char*) malloc(src_size + 1);
  if (fread(s, 1, src_size, f) != src_size) {
    free(s);
    s = NULL;
  } else {
    s[src_size] = '\0';
  }
  fclose(f);

  if (size) {
    *size = src_size;
  }

  return s;
}

// Dump 'n' bytes from 'buf' into the file at the designated location.
// Returns 0 on success.
static int dump_file(const char *file, const void *buf, size_t n) {
  FILE *f = fopen(file, "w");

  if (f == NULL) {
    return 1;
  }

  if (fwrite(buf, sizeof(char), n, f) != n) {
    return 1;
  }

  if (fclose(f) != 0) {
    return 1;
  }

  return 0;
}

struct str_builder {
  char *str;
  size_t capacity; // Size of buffer.
  size_t used; // Bytes used, *not* including final zero.
};

static void str_builder_init(struct str_builder *b) {
  b->capacity = 10;
  b->used = 0;
  b->str = malloc(b->capacity);
  b->str[0] = 0;
}

static void str_builder(struct str_builder *b, const char *s, ...) {
  va_list vl;
  va_start(vl, s);
  size_t needed = (size_t)vsnprintf(NULL, 0, s, vl);

  while (b->capacity < b->used + needed + 1) {
    b->capacity *= 2;
    b->str = realloc(b->str, b->capacity);
  }

  va_start(vl, s); // Must re-init.
  vsnprintf(b->str+b->used, b->capacity-b->used, s, vl);
  b->used += needed;
}

// End of util.h.

// Start of timing.h.

// The function get_wall_time() returns the wall time in microseconds
// (with an unspecified offset).

#ifdef _WIN32

#include <windows.h>

static int64_t get_wall_time(void) {
  LARGE_INTEGER time,freq;
  assert(QueryPerformanceFrequency(&freq));
  assert(QueryPerformanceCounter(&time));
  return ((double)time.QuadPart / freq.QuadPart) * 1000000;
}

#else
// Assuming POSIX

#include <time.h>
#include <sys/time.h>

static int64_t get_wall_time(void) {
  struct timeval time;
  assert(gettimeofday(&time,NULL) == 0);
  return time.tv_sec * 1000000 + time.tv_usec;
}

static int64_t get_wall_time_ns(void) {
  struct timespec time;
  assert(clock_gettime(CLOCK_REALTIME, &time) == 0);
  return time.tv_sec * 1000000000 + time.tv_nsec;
}

#endif

// End of timing.h.

#include <getopt.h>
#include <ctype.h>
#include <inttypes.h>
// Start of values.h.

//// Text I/O

typedef int (*writer)(FILE*, const void*);
typedef int (*bin_reader)(void*);
typedef int (*str_reader)(const char *, void*);

struct array_reader {
  char* elems;
  int64_t n_elems_space;
  int64_t elem_size;
  int64_t n_elems_used;
  int64_t *shape;
  str_reader elem_reader;
};

static void skipspaces(FILE *f) {
  int c;
  do {
    c = getc(f);
  } while (isspace(c));

  if (c != EOF) {
    ungetc(c, f);
  }
}

static int constituent(char c) {
  return isalnum(c) || c == '.' || c == '-' || c == '+' || c == '_';
}

// Produces an empty token only on EOF.
static void next_token(FILE *f, char *buf, int bufsize) {
 start:
  skipspaces(f);

  int i = 0;
  while (i < bufsize) {
    int c = getc(f);
    buf[i] = (char)c;

    if (c == EOF) {
      buf[i] = 0;
      return;
    } else if (c == '-' && i == 1 && buf[0] == '-') {
      // Line comment, so skip to end of line and start over.
      for (; c != '\n' && c != EOF; c = getc(f));
      goto start;
    } else if (!constituent((char)c)) {
      if (i == 0) {
        // We permit single-character tokens that are not
        // constituents; this lets things like ']' and ',' be
        // tokens.
        buf[i+1] = 0;
        return;
      } else {
        ungetc(c, f);
        buf[i] = 0;
        return;
      }
    }

    i++;
  }

  buf[bufsize-1] = 0;
}

static int next_token_is(FILE *f, char *buf, int bufsize, const char* expected) {
  next_token(f, buf, bufsize);
  return strcmp(buf, expected) == 0;
}

static void remove_underscores(char *buf) {
  char *w = buf;

  for (char *r = buf; *r; r++) {
    if (*r != '_') {
      *w++ = *r;
    }
  }

  *w++ = 0;
}

static int read_str_elem(char *buf, struct array_reader *reader) {
  int ret;
  if (reader->n_elems_used == reader->n_elems_space) {
    reader->n_elems_space *= 2;
    reader->elems = (char*) realloc(reader->elems,
                                    (size_t)(reader->n_elems_space * reader->elem_size));
  }

  ret = reader->elem_reader(buf, reader->elems + reader->n_elems_used * reader->elem_size);

  if (ret == 0) {
    reader->n_elems_used++;
  }

  return ret;
}

static int read_str_array_elems(FILE *f,
                                char *buf, int bufsize,
                                struct array_reader *reader, int64_t dims) {
  int ret;
  int first = 1;
  char *knows_dimsize = (char*) calloc((size_t)dims, sizeof(char));
  int cur_dim = dims-1;
  int64_t *elems_read_in_dim = (int64_t*) calloc((size_t)dims, sizeof(int64_t));

  while (1) {
    next_token(f, buf, bufsize);

    if (strcmp(buf, "]") == 0) {
      if (knows_dimsize[cur_dim]) {
        if (reader->shape[cur_dim] != elems_read_in_dim[cur_dim]) {
          ret = 1;
          break;
        }
      } else {
        knows_dimsize[cur_dim] = 1;
        reader->shape[cur_dim] = elems_read_in_dim[cur_dim];
      }
      if (cur_dim == 0) {
        ret = 0;
        break;
      } else {
        cur_dim--;
        elems_read_in_dim[cur_dim]++;
      }
    } else if (strcmp(buf, ",") == 0) {
      next_token(f, buf, bufsize);
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        first = 1;
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else if (cur_dim == dims - 1) {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
      } else {
        ret = 1;
        break;
      }
    } else if (strlen(buf) == 0) {
      // EOF
      ret = 1;
      break;
    } else if (first) {
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
        first = 0;
      }
    } else {
      ret = 1;
      break;
    }
  }

  free(knows_dimsize);
  free(elems_read_in_dim);
  return ret;
}

static int read_str_empty_array(FILE *f, char *buf, int bufsize,
                                const char *type_name, int64_t *shape, int64_t dims) {
  if (strlen(buf) == 0) {
    // EOF
    return 1;
  }

  if (strcmp(buf, "empty") != 0) {
    return 1;
  }

  if (!next_token_is(f, buf, bufsize, "(")) {
    return 1;
  }

  for (int i = 0; i < dims; i++) {
    if (!next_token_is(f, buf, bufsize, "[")) {
      return 1;
    }

    next_token(f, buf, bufsize);

    if (sscanf(buf, "%"SCNu64, (uint64_t*)&shape[i]) != 1) {
      return 1;
    }

    if (!next_token_is(f, buf, bufsize, "]")) {
      return 1;
    }
  }

  if (!next_token_is(f, buf, bufsize, type_name)) {
    return 1;
  }


  if (!next_token_is(f, buf, bufsize, ")")) {
    return 1;
  }

  // Check whether the array really is empty.
  for (int i = 0; i < dims; i++) {
    if (shape[i] == 0) {
      return 0;
    }
  }

  // Not an empty array!
  return 1;
}

static int read_str_array(FILE *f,
                          int64_t elem_size, str_reader elem_reader,
                          const char *type_name,
                          void **data, int64_t *shape, int64_t dims) {
  int ret;
  struct array_reader reader;
  char buf[100];

  int dims_seen;
  for (dims_seen = 0; dims_seen < dims; dims_seen++) {
    if (!next_token_is(f, buf, sizeof(buf), "[")) {
      break;
    }
  }

  if (dims_seen == 0) {
    return read_str_empty_array(f, buf, sizeof(buf), type_name, shape, dims);
  }

  if (dims_seen != dims) {
    return 1;
  }

  reader.shape = shape;
  reader.n_elems_used = 0;
  reader.elem_size = elem_size;
  reader.n_elems_space = 16;
  reader.elems = (char*) realloc(*data, (size_t)(elem_size*reader.n_elems_space));
  reader.elem_reader = elem_reader;

  ret = read_str_array_elems(f, buf, sizeof(buf), &reader, dims);

  *data = reader.elems;

  return ret;
}

#define READ_STR(MACRO, PTR, SUFFIX)                                   \
  remove_underscores(buf);                                              \
  int j;                                                                \
  if (sscanf(buf, "%"MACRO"%n", (PTR*)dest, &j) == 1) {                 \
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, SUFFIX) == 0);     \
  } else {                                                              \
    return 1;                                                           \
  }

static int read_str_i8(char *buf, void* dest) {
  // Some platforms (WINDOWS) does not support scanf %hhd or its
  // cousin, %SCNi8.  Read into int first to avoid corrupting
  // memory.
  //
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(int8_t*)dest = (int8_t)x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "i8") == 0);
  } else {
    return 1;
  }
}

static int read_str_u8(char *buf, void* dest) {
  // Some platforms (WINDOWS) does not support scanf %hhd or its
  // cousin, %SCNu8.  Read into int first to avoid corrupting
  // memory.
  //
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(uint8_t*)dest = (uint8_t)x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "u8") == 0);
  } else {
    return 1;
  }
}

static int read_str_i16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "i16");
}

static int read_str_u16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "u16");
}

static int read_str_i32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "i32");
}

static int read_str_u32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "u32");
}

static int read_str_i64(char *buf, void* dest) {
  READ_STR(SCNi64, int64_t, "i64");
}

static int read_str_u64(char *buf, void* dest) {
  // FIXME: This is not correct, as SCNu64 only permits decimal
  // literals.  However, SCNi64 does not handle very large numbers
  // correctly (it's really for signed numbers, so that's fair).
  READ_STR(SCNu64, uint64_t, "u64");
}

static int read_str_f32(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f32.nan") == 0) {
    *(float*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f32.inf") == 0) {
    *(float*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f32.inf") == 0) {
    *(float*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("f", float, "f32");
  }
}

static int read_str_f64(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f64.nan") == 0) {
    *(double*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f64.inf") == 0) {
    *(double*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f64.inf") == 0) {
    *(double*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("lf", double, "f64");
  }
}

static int read_str_bool(char *buf, void* dest) {
  if (strcmp(buf, "true") == 0) {
    *(char*)dest = 1;
    return 0;
  } else if (strcmp(buf, "false") == 0) {
    *(char*)dest = 0;
    return 0;
  } else {
    return 1;
  }
}

static int write_str_i8(FILE *out, int8_t *src) {
  return fprintf(out, "%hhdi8", *src);
}

static int write_str_u8(FILE *out, uint8_t *src) {
  return fprintf(out, "%hhuu8", *src);
}

static int write_str_i16(FILE *out, int16_t *src) {
  return fprintf(out, "%hdi16", *src);
}

static int write_str_u16(FILE *out, uint16_t *src) {
  return fprintf(out, "%huu16", *src);
}

static int write_str_i32(FILE *out, int32_t *src) {
  return fprintf(out, "%di32", *src);
}

static int write_str_u32(FILE *out, uint32_t *src) {
  return fprintf(out, "%uu32", *src);
}

static int write_str_i64(FILE *out, int64_t *src) {
  return fprintf(out, "%"PRIi64"i64", *src);
}

static int write_str_u64(FILE *out, uint64_t *src) {
  return fprintf(out, "%"PRIu64"u64", *src);
}

static int write_str_f32(FILE *out, float *src) {
  float x = *src;
  if (isnan(x)) {
    return fprintf(out, "f32.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f32.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f32.inf");
  } else {
    return fprintf(out, "%.6ff32", x);
  }
}

static int write_str_f64(FILE *out, double *src) {
  double x = *src;
  if (isnan(x)) {
    return fprintf(out, "f64.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f64.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f64.inf");
  } else {
    return fprintf(out, "%.6ff64", *src);
  }
}

static int write_str_bool(FILE *out, void *src) {
  return fprintf(out, *(char*)src ? "true" : "false");
}

//// Binary I/O

#define BINARY_FORMAT_VERSION 2
#define IS_BIG_ENDIAN (!*(unsigned char *)&(uint16_t){1})

static void flip_bytes(int elem_size, unsigned char *elem) {
  for (int j=0; j<elem_size/2; j++) {
    unsigned char head = elem[j];
    int tail_index = elem_size-1-j;
    elem[j] = elem[tail_index];
    elem[tail_index] = head;
  }
}

// On Windows we need to explicitly set the file mode to not mangle
// newline characters.  On *nix there is no difference.
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
static void set_binary_mode(FILE *f) {
  setmode(fileno(f), O_BINARY);
}
#else
static void set_binary_mode(FILE *f) {
  (void)f;
}
#endif

static int read_byte(FILE *f, void* dest) {
  int num_elems_read = fread(dest, 1, 1, f);
  return num_elems_read == 1 ? 0 : 1;
}

//// Types

struct primtype_info_t {
  const char binname[4]; // Used for parsing binary data.
  const char* type_name; // Same name as in Futhark.
  const int64_t size; // in bytes
  const writer write_str; // Write in text format.
  const str_reader read_str; // Read in text format.
};

static const struct primtype_info_t i8_info =
  {.binname = "  i8", .type_name = "i8",   .size = 1,
   .write_str = (writer)write_str_i8, .read_str = (str_reader)read_str_i8};
static const struct primtype_info_t i16_info =
  {.binname = " i16", .type_name = "i16",  .size = 2,
   .write_str = (writer)write_str_i16, .read_str = (str_reader)read_str_i16};
static const struct primtype_info_t i32_info =
  {.binname = " i32", .type_name = "i32",  .size = 4,
   .write_str = (writer)write_str_i32, .read_str = (str_reader)read_str_i32};
static const struct primtype_info_t i64_info =
  {.binname = " i64", .type_name = "i64",  .size = 8,
   .write_str = (writer)write_str_i64, .read_str = (str_reader)read_str_i64};
static const struct primtype_info_t u8_info =
  {.binname = "  u8", .type_name = "u8",   .size = 1,
   .write_str = (writer)write_str_u8, .read_str = (str_reader)read_str_u8};
static const struct primtype_info_t u16_info =
  {.binname = " u16", .type_name = "u16",  .size = 2,
   .write_str = (writer)write_str_u16, .read_str = (str_reader)read_str_u16};
static const struct primtype_info_t u32_info =
  {.binname = " u32", .type_name = "u32",  .size = 4,
   .write_str = (writer)write_str_u32, .read_str = (str_reader)read_str_u32};
static const struct primtype_info_t u64_info =
  {.binname = " u64", .type_name = "u64",  .size = 8,
   .write_str = (writer)write_str_u64, .read_str = (str_reader)read_str_u64};
static const struct primtype_info_t f32_info =
  {.binname = " f32", .type_name = "f32",  .size = 4,
   .write_str = (writer)write_str_f32, .read_str = (str_reader)read_str_f32};
static const struct primtype_info_t f64_info =
  {.binname = " f64", .type_name = "f64",  .size = 8,
   .write_str = (writer)write_str_f64, .read_str = (str_reader)read_str_f64};
static const struct primtype_info_t bool_info =
  {.binname = "bool", .type_name = "bool", .size = 1,
   .write_str = (writer)write_str_bool, .read_str = (str_reader)read_str_bool};

static const struct primtype_info_t* primtypes[] = {
  &i8_info, &i16_info, &i32_info, &i64_info,
  &u8_info, &u16_info, &u32_info, &u64_info,
  &f32_info, &f64_info,
  &bool_info,
  NULL // NULL-terminated
};

// General value interface.  All endian business taken care of at
// lower layers.

static int read_is_binary(FILE *f) {
  skipspaces(f);
  int c = getc(f);
  if (c == 'b') {
    int8_t bin_version;
    int ret = read_byte(f, &bin_version);

    if (ret != 0) { futhark_panic(1, "binary-input: could not read version.\n"); }

    if (bin_version != BINARY_FORMAT_VERSION) {
      futhark_panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
            bin_version, BINARY_FORMAT_VERSION);
    }

    return 1;
  }
  ungetc(c, f);
  return 0;
}

static const struct primtype_info_t* read_bin_read_type_enum(FILE *f) {
  char read_binname[4];

  int num_matched = fscanf(f, "%4c", read_binname);
  if (num_matched != 1) { futhark_panic(1, "binary-input: Couldn't read element type.\n"); }

  const struct primtype_info_t **type = primtypes;

  for (; *type != NULL; type++) {
    // I compare the 4 characters manually instead of using strncmp because
    // this allows any value to be used, also NULL bytes
    if (memcmp(read_binname, (*type)->binname, 4) == 0) {
      return *type;
    }
  }
  futhark_panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname);
  return NULL;
}

static void read_bin_ensure_scalar(FILE *f, const struct primtype_info_t *expected_type) {
  int8_t bin_dims;
  int ret = read_byte(f, &bin_dims);
  if (ret != 0) { futhark_panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != 0) {
    futhark_panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n",
          bin_dims);
  }

  const struct primtype_info_t *bin_type = read_bin_read_type_enum(f);
  if (bin_type != expected_type) {
    futhark_panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
          expected_type->type_name,
          bin_type->type_name);
  }
}

//// High-level interface

static int read_bin_array(FILE *f,
                          const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  int ret;

  int8_t bin_dims;
  ret = read_byte(f, &bin_dims);
  if (ret != 0) { futhark_panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != dims) {
    futhark_panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
          dims, bin_dims);
  }

  const struct primtype_info_t *bin_primtype = read_bin_read_type_enum(f);
  if (expected_type != bin_primtype) {
    futhark_panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
          dims, expected_type->type_name, dims, bin_primtype->type_name);
  }

  int64_t elem_count = 1;
  for (int i=0; i<dims; i++) {
    int64_t bin_shape;
    ret = fread(&bin_shape, sizeof(bin_shape), 1, f);
    if (ret != 1) {
      futhark_panic(1, "binary-input: Couldn't read size for dimension %i of array.\n", i);
    }
    if (IS_BIG_ENDIAN) {
      flip_bytes(sizeof(bin_shape), (unsigned char*) &bin_shape);
    }
    elem_count *= bin_shape;
    shape[i] = bin_shape;
  }

  int64_t elem_size = expected_type->size;
  void* tmp = realloc(*data, (size_t)(elem_count * elem_size));
  if (tmp == NULL) {
    futhark_panic(1, "binary-input: Failed to allocate array of size %i.\n",
          elem_count * elem_size);
  }
  *data = tmp;

  int64_t num_elems_read = (int64_t)fread(*data, (size_t)elem_size, (size_t)elem_count, f);
  if (num_elems_read != elem_count) {
    futhark_panic(1, "binary-input: tried to read %i elements of an array, but only got %i elements.\n",
          elem_count, num_elems_read);
  }

  // If we're on big endian platform we must change all multibyte elements
  // from using little endian to big endian
  if (IS_BIG_ENDIAN && elem_size != 1) {
    flip_bytes(elem_size, (unsigned char*) *data);
  }

  return 0;
}

static int read_array(FILE *f, const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  if (!read_is_binary(f)) {
    return read_str_array(f, expected_type->size, (str_reader)expected_type->read_str, expected_type->type_name, data, shape, dims);
  } else {
    return read_bin_array(f, expected_type, data, shape, dims);
  }
}

static int end_of_input(FILE *f) {
  skipspaces(f);
  char token[2];
  next_token(f, token, sizeof(token));
  if (strcmp(token, "") == 0) {
    return 0;
  } else {
    return 1;
  }
}

static int write_str_array(FILE *out,
                           const struct primtype_info_t *elem_type,
                           const unsigned char *data,
                           const int64_t *shape,
                           int8_t rank) {
  if (rank==0) {
    elem_type->write_str(out, (void*)data);
  } else {
    int64_t len = (int64_t)shape[0];
    int64_t slice_size = 1;

    int64_t elem_size = elem_type->size;
    for (int8_t i = 1; i < rank; i++) {
      slice_size *= shape[i];
    }

    if (len*slice_size == 0) {
      fprintf(out, "empty(");
      for (int64_t i = 0; i < rank; i++) {
        fprintf(out, "[%"PRIi64"]", shape[i]);
      }
      fprintf(out, "%s", elem_type->type_name);
      fprintf(out, ")");
    } else if (rank==1) {
      fputc('[', out);
      for (int64_t i = 0; i < len; i++) {
        elem_type->write_str(out, (void*) (data + i * elem_size));
        if (i != len-1) {
          fprintf(out, ", ");
        }
      }
      fputc(']', out);
    } else {
      fputc('[', out);
      for (int64_t i = 0; i < len; i++) {
        write_str_array(out, elem_type, data + i * slice_size * elem_size, shape+1, rank-1);
        if (i != len-1) {
          fprintf(out, ", ");
        }
      }
      fputc(']', out);
    }
  }
  return 0;
}

static int write_bin_array(FILE *out,
                           const struct primtype_info_t *elem_type,
                           const unsigned char *data,
                           const int64_t *shape,
                           int8_t rank) {
  int64_t num_elems = 1;
  for (int64_t i = 0; i < rank; i++) {
    num_elems *= shape[i];
  }

  fputc('b', out);
  fputc((char)BINARY_FORMAT_VERSION, out);
  fwrite(&rank, sizeof(int8_t), 1, out);
  fwrite(elem_type->binname, 4, 1, out);
  if (shape != NULL) {
    fwrite(shape, sizeof(int64_t), (size_t)rank, out);
  }

  if (IS_BIG_ENDIAN) {
    for (int64_t i = 0; i < num_elems; i++) {
      const unsigned char *elem = data+i*elem_type->size;
      for (int64_t j = 0; j < elem_type->size; j++) {
        fwrite(&elem[elem_type->size-j], 1, 1, out);
      }
    }
  } else {
    fwrite(data, (size_t)elem_type->size, (size_t)num_elems, out);
  }

  return 0;
}

static int write_array(FILE *out, int write_binary,
                       const struct primtype_info_t *elem_type,
                       const void *data,
                       const int64_t *shape,
                       const int8_t rank) {
  if (write_binary) {
    return write_bin_array(out, elem_type, data, shape, rank);
  } else {
    return write_str_array(out, elem_type, data, shape, rank);
  }
}

static int read_scalar(FILE *f,
                       const struct primtype_info_t *expected_type, void *dest) {
  if (!read_is_binary(f)) {
    char buf[100];
    next_token(f, buf, sizeof(buf));
    return expected_type->read_str(buf, dest);
  } else {
    read_bin_ensure_scalar(f, expected_type);
    int64_t elem_size = expected_type->size;
    int num_elems_read = fread(dest, (size_t)elem_size, 1, f);
    if (IS_BIG_ENDIAN) {
      flip_bytes(elem_size, (unsigned char*) dest);
    }
    return num_elems_read == 1 ? 0 : 1;
  }
}

static int write_scalar(FILE *out, int write_binary, const struct primtype_info_t *type, void *src) {
  if (write_binary) {
    return write_bin_array(out, type, src, NULL, 0);
  } else {
    return type->write_str(out, src);
  }
}

// End of values.h.

static int binary_output = 0;
static FILE *runtime_file;
static int perform_warmup = 0;
static int num_runs = 1;
static const char *entry_point = "main";
// Start of tuning.h.

static char* load_tuning_file(const char *fname,
                              void *cfg,
                              int (*set_size)(void*, const char*, size_t)) {
  const int max_line_len = 1024;
  char* line = (char*) malloc(max_line_len);

  FILE *f = fopen(fname, "r");

  if (f == NULL) {
    snprintf(line, max_line_len, "Cannot open file: %s", strerror(errno));
    return line;
  }

  int lineno = 0;
  while (fgets(line, max_line_len, f) != NULL) {
    lineno++;
    char *eql = strstr(line, "=");
    if (eql) {
      *eql = 0;
      int value = atoi(eql+1);
      if (set_size(cfg, line, value) != 0) {
        strncpy(eql+1, line, max_line_len-strlen(line)-1);
        snprintf(line, max_line_len, "Unknown name '%s' on line %d.", eql+1, lineno);
        return line;
      }
    } else {
      snprintf(line, max_line_len, "Invalid line %d (must be of form 'name=int').",
               lineno);
      return line;
    }
  }

  free(line);

  return NULL;
}

// End of tuning.h.

int parse_options(struct futhark_context_config *cfg, int argc,
                  char *const argv[])
{
    int ch;
    static struct option long_options[] = {{"write-runtime-to",
                                            required_argument, NULL, 1},
                                           {"runs", required_argument, NULL, 2},
                                           {"debugging", no_argument, NULL, 3},
                                           {"log", no_argument, NULL, 4},
                                           {"entry-point", required_argument,
                                            NULL, 5}, {"binary-output",
                                                       no_argument, NULL, 6},
                                           {"help", no_argument, NULL, 7},
                                           {"device", required_argument, NULL,
                                            8}, {"default-group-size",
                                                 required_argument, NULL, 9},
                                           {"default-num-groups",
                                            required_argument, NULL, 10},
                                           {"default-tile-size",
                                            required_argument, NULL, 11},
                                           {"default-threshold",
                                            required_argument, NULL, 12},
                                           {"print-sizes", no_argument, NULL,
                                            13}, {"size", required_argument,
                                                  NULL, 14}, {"tuning",
                                                              required_argument,
                                                              NULL, 15},
                                           {"platform", required_argument, NULL,
                                            16}, {"dump-opencl",
                                                  required_argument, NULL, 17},
                                           {"load-opencl", required_argument,
                                            NULL, 18}, {"dump-opencl-binary",
                                                        required_argument, NULL,
                                                        19},
                                           {"load-opencl-binary",
                                            required_argument, NULL, 20},
                                           {"build-option", required_argument,
                                            NULL, 21}, {"profile", no_argument,
                                                        NULL, 22},
                                           {"list-devices", no_argument, NULL,
                                            23}, {0, 0, 0, 0}};
    static char *option_descriptions =
                "  -t/--write-runtime-to FILE Print the time taken to execute the program to the indicated file, an integral number of microseconds.\n  -r/--runs INT              Perform NUM runs of the program.\n  -D/--debugging             Perform possibly expensive internal correctness checks and verbose logging.\n  -L/--log                   Print various low-overhead logging information to stderr while running.\n  -e/--entry-point NAME      The entry point to run. Defaults to main.\n  -b/--binary-output         Print the program result in the binary output format.\n  -h/--help                  Print help information and exit.\n  -d/--device NAME           Use the first OpenCL device whose name contains the given string.\n  --default-group-size INT   The default size of OpenCL workgroups that are launched.\n  --default-num-groups INT   The default number of OpenCL workgroups that are launched.\n  --default-tile-size INT    The default tile size used when performing two-dimensional tiling.\n  --default-threshold INT    The default parallelism threshold.\n  --print-sizes              Print all sizes that can be set with -size or --tuning.\n  --size ASSIGNMENT          Set a configurable run-time parameter to the given value.\n  --tuning FILE              Read size=value assignments from the given file.\n  -p/--platform NAME         Use the first OpenCL platform whose name contains the given string.\n  --dump-opencl FILE         Dump the embedded OpenCL program to the indicated file.\n  --load-opencl FILE         Instead of using the embedded OpenCL program, load it from the indicated file.\n  --dump-opencl-binary FILE  Dump the compiled version of the embedded OpenCL program to the indicated file.\n  --load-opencl-binary FILE  Load an OpenCL binary from the indicated file.\n  --build-option OPT         Add an additional build option to the string passed to clBuildProgram().\n  -P/--profile               Gather profiling data while executing and print out a summary at the end.\n  --list-devices             List all OpenCL devices and platforms available on the system.\n";
    
    while ((ch = getopt_long(argc, argv, ":t:r:DLe:bhd:p:P", long_options,
                             NULL)) != -1) {
        if (ch == 1 || ch == 't') {
            runtime_file = fopen(optarg, "w");
            if (runtime_file == NULL)
                futhark_panic(1, "Cannot open %s: %s\n", optarg,
                              strerror(errno));
        }
        if (ch == 2 || ch == 'r') {
            num_runs = atoi(optarg);
            perform_warmup = 1;
            if (num_runs <= 0)
                futhark_panic(1, "Need a positive number of runs, not %s\n",
                              optarg);
        }
        if (ch == 3 || ch == 'D')
            futhark_context_config_set_debugging(cfg, 1);
        if (ch == 4 || ch == 'L')
            futhark_context_config_set_logging(cfg, 1);
        if (ch == 5 || ch == 'e') {
            if (entry_point != NULL)
                entry_point = optarg;
        }
        if (ch == 6 || ch == 'b')
            binary_output = 1;
        if (ch == 7 || ch == 'h') {
            printf("Usage: %s [OPTION]...\nOptions:\n\n%s\nFor more information, consult the Futhark User's Guide or the man pages.\n",
                   fut_progname, option_descriptions);
            exit(0);
        }
        if (ch == 8 || ch == 'd')
            futhark_context_config_set_device(cfg, optarg);
        if (ch == 9)
            futhark_context_config_set_default_group_size(cfg, atoi(optarg));
        if (ch == 10)
            futhark_context_config_set_default_num_groups(cfg, atoi(optarg));
        if (ch == 11)
            futhark_context_config_set_default_tile_size(cfg, atoi(optarg));
        if (ch == 12)
            futhark_context_config_set_default_threshold(cfg, atoi(optarg));
        if (ch == 13) {
            int n = futhark_get_num_sizes();
            
            for (int i = 0; i < n; i++)
                printf("%s (%s)\n", futhark_get_size_name(i),
                       futhark_get_size_class(i));
            exit(0);
        }
        if (ch == 14) {
            char *name = optarg;
            char *equals = strstr(optarg, "=");
            char *value_str = equals != NULL ? equals + 1 : optarg;
            int value = atoi(value_str);
            
            if (equals != NULL) {
                *equals = 0;
                if (futhark_context_config_set_size(cfg, name, value) != 0)
                    futhark_panic(1, "Unknown size: %s\n", name);
            } else
                futhark_panic(1, "Invalid argument for size option: %s\n",
                              optarg);
        }
        if (ch == 15) {
            char *ret = load_tuning_file(optarg, cfg, (int (*)(void *, const
                                                               char *,
                                                               size_t)) futhark_context_config_set_size);
            
            if (ret != NULL)
                futhark_panic(1, "When loading tuning from '%s': %s\n", optarg,
                              ret);
        }
        if (ch == 16 || ch == 'p')
            futhark_context_config_set_platform(cfg, optarg);
        if (ch == 17) {
            futhark_context_config_dump_program_to(cfg, optarg);
            entry_point = NULL;
        }
        if (ch == 18)
            futhark_context_config_load_program_from(cfg, optarg);
        if (ch == 19) {
            futhark_context_config_dump_binary_to(cfg, optarg);
            entry_point = NULL;
        }
        if (ch == 20)
            futhark_context_config_load_binary_from(cfg, optarg);
        if (ch == 21)
            futhark_context_config_add_build_option(cfg, optarg);
        if (ch == 22 || ch == 'P')
            futhark_context_config_set_profiling(cfg, 1);
        if (ch == 23) {
            futhark_context_config_list_devices(cfg);
            entry_point = NULL;
        }
        if (ch == ':')
            futhark_panic(-1, "Missing argument for option %s\n", argv[optind -
                                                                       1]);
        if (ch == '?') {
            fprintf(stderr, "Usage: %s: %s\n", fut_progname,
                    "  -t/--write-runtime-to FILE Print the time taken to execute the program to the indicated file, an integral number of microseconds.\n  -r/--runs INT              Perform NUM runs of the program.\n  -D/--debugging             Perform possibly expensive internal correctness checks and verbose logging.\n  -L/--log                   Print various low-overhead logging information to stderr while running.\n  -e/--entry-point NAME      The entry point to run. Defaults to main.\n  -b/--binary-output         Print the program result in the binary output format.\n  -h/--help                  Print help information and exit.\n  -d/--device NAME           Use the first OpenCL device whose name contains the given string.\n  --default-group-size INT   The default size of OpenCL workgroups that are launched.\n  --default-num-groups INT   The default number of OpenCL workgroups that are launched.\n  --default-tile-size INT    The default tile size used when performing two-dimensional tiling.\n  --default-threshold INT    The default parallelism threshold.\n  --print-sizes              Print all sizes that can be set with -size or --tuning.\n  --size ASSIGNMENT          Set a configurable run-time parameter to the given value.\n  --tuning FILE              Read size=value assignments from the given file.\n  -p/--platform NAME         Use the first OpenCL platform whose name contains the given string.\n  --dump-opencl FILE         Dump the embedded OpenCL program to the indicated file.\n  --load-opencl FILE         Instead of using the embedded OpenCL program, load it from the indicated file.\n  --dump-opencl-binary FILE  Dump the compiled version of the embedded OpenCL program to the indicated file.\n  --load-opencl-binary FILE  Load an OpenCL binary from the indicated file.\n  --build-option OPT         Add an additional build option to the string passed to clBuildProgram().\n  -P/--profile               Gather profiling data while executing and print out a summary at the end.\n  --list-devices             List all OpenCL devices and platforms available on the system.\n");
            futhark_panic(1, "Unknown option: %s\n", argv[optind - 1]);
        }
    }
    return optind;
}
static void futrts_cli_entry_main(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    // Declare and read input.
    set_binary_mode(stdin);
    
    struct futhark_i32_1d *read_value_0;
    int64_t read_shape_0[1];
    int32_t *read_arr_0 = NULL;
    
    errno = 0;
    if (read_array(stdin, &i32_info, (void **) &read_arr_0, read_shape_0, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0,
                      "[]", i32_info.type_name, strerror(errno));
    if (end_of_input(stdin) != 0)
        futhark_panic(1, "Expected EOF on stdin after reading input for %s.\n",
                      "\"main\"");
    
    struct futhark_i32_1d *result_0;
    
    if (perform_warmup) {
        int r;
        
        assert((read_value_0 = futhark_new_i32_1d(ctx, read_arr_0,
                                                  read_shape_0[0])) != NULL);
        if (futhark_context_sync(ctx) != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        ;
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_0, read_value_0);
        if (r != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        if (futhark_context_sync(ctx) != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        ;
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL) {
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
            fflush(runtime_file);
        }
        assert(futhark_free_i32_1d(ctx, read_value_0) == 0);
        assert(futhark_free_i32_1d(ctx, result_0) == 0);
    }
    time_runs = 1;
    // Proper run.
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        assert((read_value_0 = futhark_new_i32_1d(ctx, read_arr_0,
                                                  read_shape_0[0])) != NULL);
        if (futhark_context_sync(ctx) != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        ;
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_0, read_value_0);
        if (r != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        if (futhark_context_sync(ctx) != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        ;
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL) {
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
            fflush(runtime_file);
        }
        assert(futhark_free_i32_1d(ctx, read_value_0) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_i32_1d(ctx, result_0) == 0);
        }
    }
    free(read_arr_0);
    if (binary_output)
        set_binary_mode(stdout);
    {
        int32_t *arr = calloc(sizeof(int32_t), futhark_shape_i32_1d(ctx,
                                                                    result_0)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_i32_1d(ctx, result_0, arr) == 0);
        write_array(stdout, binary_output, &i32_info, arr,
                    futhark_shape_i32_1d(ctx, result_0), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_i32_1d(ctx, result_0) == 0);
}
typedef void entry_point_fun(struct futhark_context *);
struct entry_point_entry {
    const char *name;
    entry_point_fun *fun;
} ;
int main(int argc, char **argv)
{
    fut_progname = argv[0];
    
    struct futhark_context_config *cfg = futhark_context_config_new();
    
    assert(cfg != NULL);
    
    int parsed_options = parse_options(cfg, argc, argv);
    
    argc -= parsed_options;
    argv += parsed_options;
    if (argc != 0)
        futhark_panic(1, "Excess non-option: %s\n", argv[0]);
    
    struct futhark_context *ctx = futhark_context_new(cfg);
    
    assert(ctx != NULL);
    
    char *error = futhark_context_get_error(ctx);
    
    if (error != NULL)
        futhark_panic(1, "%s", error);
    
    struct entry_point_entry entry_points[] = {{.name ="main", .fun =
                                                futrts_cli_entry_main}};
    
    if (entry_point != NULL) {
        int num_entry_points = sizeof(entry_points) / sizeof(entry_points[0]);
        entry_point_fun *entry_point_fun = NULL;
        
        for (int i = 0; i < num_entry_points; i++) {
            if (strcmp(entry_points[i].name, entry_point) == 0) {
                entry_point_fun = entry_points[i].fun;
                break;
            }
        }
        if (entry_point_fun == NULL) {
            fprintf(stderr,
                    "No entry point '%s'.  Select another with --entry-point.  Options are:\n",
                    entry_point);
            for (int i = 0; i < num_entry_points; i++)
                fprintf(stderr, "%s\n", entry_points[i].name);
            return 1;
        }
        entry_point_fun(ctx);
        if (runtime_file != NULL)
            fclose(runtime_file);
        
        char *report = futhark_context_report(ctx);
        
        fputs(report, stderr);
        free(report);
    }
    futhark_context_free(ctx);
    futhark_context_config_free(cfg);
    return 0;
}
#ifdef _MSC_VER
#define inline __inline
#endif
#include <string.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#define CL_SILENCE_DEPRECATION
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// Start of lock.h.

// A very simple cross-platform implementation of locks.  Uses
// pthreads on Unix and some Windows thing there.  Futhark's
// host-level code is not multithreaded, but user code may be, so we
// need some mechanism for ensuring atomic access to API functions.
// This is that mechanism.  It is not exposed to user code at all, so
// we do not have to worry about name collisions.

#ifdef _WIN32

typedef HANDLE lock_t;

static void create_lock(lock_t *lock) {
  *lock = CreateMutex(NULL,  // Default security attributes.
                      FALSE, // Initially unlocked.
                      NULL); // Unnamed.
}

static void lock_lock(lock_t *lock) {
  assert(WaitForSingleObject(*lock, INFINITE) == WAIT_OBJECT_0);
}

static void lock_unlock(lock_t *lock) {
  assert(ReleaseMutex(*lock));
}

static void free_lock(lock_t *lock) {
  CloseHandle(*lock);
}

#else
// Assuming POSIX

#include <pthread.h>

typedef pthread_mutex_t lock_t;

static void create_lock(lock_t *lock) {
  int r = pthread_mutex_init(lock, NULL);
  assert(r == 0);
}

static void lock_lock(lock_t *lock) {
  int r = pthread_mutex_lock(lock);
  assert(r == 0);
}

static void lock_unlock(lock_t *lock) {
  int r = pthread_mutex_unlock(lock);
  assert(r == 0);
}

static void free_lock(lock_t *lock) {
  // Nothing to do for pthreads.
  (void)lock;
}

#endif

// End of lock.h.

static inline uint8_t add8(uint8_t x, uint8_t y)
{
    return x + y;
}
static inline uint16_t add16(uint16_t x, uint16_t y)
{
    return x + y;
}
static inline uint32_t add32(uint32_t x, uint32_t y)
{
    return x + y;
}
static inline uint64_t add64(uint64_t x, uint64_t y)
{
    return x + y;
}
static inline uint8_t sub8(uint8_t x, uint8_t y)
{
    return x - y;
}
static inline uint16_t sub16(uint16_t x, uint16_t y)
{
    return x - y;
}
static inline uint32_t sub32(uint32_t x, uint32_t y)
{
    return x - y;
}
static inline uint64_t sub64(uint64_t x, uint64_t y)
{
    return x - y;
}
static inline uint8_t mul8(uint8_t x, uint8_t y)
{
    return x * y;
}
static inline uint16_t mul16(uint16_t x, uint16_t y)
{
    return x * y;
}
static inline uint32_t mul32(uint32_t x, uint32_t y)
{
    return x * y;
}
static inline uint64_t mul64(uint64_t x, uint64_t y)
{
    return x * y;
}
static inline uint8_t udiv8(uint8_t x, uint8_t y)
{
    return x / y;
}
static inline uint16_t udiv16(uint16_t x, uint16_t y)
{
    return x / y;
}
static inline uint32_t udiv32(uint32_t x, uint32_t y)
{
    return x / y;
}
static inline uint64_t udiv64(uint64_t x, uint64_t y)
{
    return x / y;
}
static inline uint8_t udiv_up8(uint8_t x, uint8_t y)
{
    return (x + y - 1) / y;
}
static inline uint16_t udiv_up16(uint16_t x, uint16_t y)
{
    return (x + y - 1) / y;
}
static inline uint32_t udiv_up32(uint32_t x, uint32_t y)
{
    return (x + y - 1) / y;
}
static inline uint64_t udiv_up64(uint64_t x, uint64_t y)
{
    return (x + y - 1) / y;
}
static inline uint8_t umod8(uint8_t x, uint8_t y)
{
    return x % y;
}
static inline uint16_t umod16(uint16_t x, uint16_t y)
{
    return x % y;
}
static inline uint32_t umod32(uint32_t x, uint32_t y)
{
    return x % y;
}
static inline uint64_t umod64(uint64_t x, uint64_t y)
{
    return x % y;
}
static inline uint8_t udiv_safe8(uint8_t x, uint8_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint16_t udiv_safe16(uint16_t x, uint16_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint32_t udiv_safe32(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint64_t udiv_safe64(uint64_t x, uint64_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint8_t udiv_up_safe8(uint8_t x, uint8_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint16_t udiv_up_safe16(uint16_t x, uint16_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint32_t udiv_up_safe32(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint64_t udiv_up_safe64(uint64_t x, uint64_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint8_t umod_safe8(uint8_t x, uint8_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline uint16_t umod_safe16(uint16_t x, uint16_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline uint32_t umod_safe32(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline uint64_t umod_safe64(uint64_t x, uint64_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int8_t sdiv8(int8_t x, int8_t y)
{
    int8_t q = x / y;
    int8_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int16_t sdiv16(int16_t x, int16_t y)
{
    int16_t q = x / y;
    int16_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int32_t sdiv32(int32_t x, int32_t y)
{
    int32_t q = x / y;
    int32_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int64_t sdiv64(int64_t x, int64_t y)
{
    int64_t q = x / y;
    int64_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int8_t sdiv_up8(int8_t x, int8_t y)
{
    return sdiv8(x + y - 1, y);
}
static inline int16_t sdiv_up16(int16_t x, int16_t y)
{
    return sdiv16(x + y - 1, y);
}
static inline int32_t sdiv_up32(int32_t x, int32_t y)
{
    return sdiv32(x + y - 1, y);
}
static inline int64_t sdiv_up64(int64_t x, int64_t y)
{
    return sdiv64(x + y - 1, y);
}
static inline int8_t smod8(int8_t x, int8_t y)
{
    int8_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int16_t smod16(int16_t x, int16_t y)
{
    int16_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int32_t smod32(int32_t x, int32_t y)
{
    int32_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int64_t smod64(int64_t x, int64_t y)
{
    int64_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int8_t sdiv_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : sdiv8(x, y);
}
static inline int16_t sdiv_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : sdiv16(x, y);
}
static inline int32_t sdiv_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : sdiv32(x, y);
}
static inline int64_t sdiv_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : sdiv64(x, y);
}
static inline int8_t sdiv_up_safe8(int8_t x, int8_t y)
{
    return sdiv_safe8(x + y - 1, y);
}
static inline int16_t sdiv_up_safe16(int16_t x, int16_t y)
{
    return sdiv_safe16(x + y - 1, y);
}
static inline int32_t sdiv_up_safe32(int32_t x, int32_t y)
{
    return sdiv_safe32(x + y - 1, y);
}
static inline int64_t sdiv_up_safe64(int64_t x, int64_t y)
{
    return sdiv_safe64(x + y - 1, y);
}
static inline int8_t smod_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : smod8(x, y);
}
static inline int16_t smod_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : smod16(x, y);
}
static inline int32_t smod_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : smod32(x, y);
}
static inline int64_t smod_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : smod64(x, y);
}
static inline int8_t squot8(int8_t x, int8_t y)
{
    return x / y;
}
static inline int16_t squot16(int16_t x, int16_t y)
{
    return x / y;
}
static inline int32_t squot32(int32_t x, int32_t y)
{
    return x / y;
}
static inline int64_t squot64(int64_t x, int64_t y)
{
    return x / y;
}
static inline int8_t srem8(int8_t x, int8_t y)
{
    return x % y;
}
static inline int16_t srem16(int16_t x, int16_t y)
{
    return x % y;
}
static inline int32_t srem32(int32_t x, int32_t y)
{
    return x % y;
}
static inline int64_t srem64(int64_t x, int64_t y)
{
    return x % y;
}
static inline int8_t squot_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int16_t squot_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int32_t squot_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int64_t squot_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int8_t srem_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int16_t srem_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int32_t srem_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int64_t srem_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int8_t smin8(int8_t x, int8_t y)
{
    return x < y ? x : y;
}
static inline int16_t smin16(int16_t x, int16_t y)
{
    return x < y ? x : y;
}
static inline int32_t smin32(int32_t x, int32_t y)
{
    return x < y ? x : y;
}
static inline int64_t smin64(int64_t x, int64_t y)
{
    return x < y ? x : y;
}
static inline uint8_t umin8(uint8_t x, uint8_t y)
{
    return x < y ? x : y;
}
static inline uint16_t umin16(uint16_t x, uint16_t y)
{
    return x < y ? x : y;
}
static inline uint32_t umin32(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}
static inline uint64_t umin64(uint64_t x, uint64_t y)
{
    return x < y ? x : y;
}
static inline int8_t smax8(int8_t x, int8_t y)
{
    return x < y ? y : x;
}
static inline int16_t smax16(int16_t x, int16_t y)
{
    return x < y ? y : x;
}
static inline int32_t smax32(int32_t x, int32_t y)
{
    return x < y ? y : x;
}
static inline int64_t smax64(int64_t x, int64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t umax8(uint8_t x, uint8_t y)
{
    return x < y ? y : x;
}
static inline uint16_t umax16(uint16_t x, uint16_t y)
{
    return x < y ? y : x;
}
static inline uint32_t umax32(uint32_t x, uint32_t y)
{
    return x < y ? y : x;
}
static inline uint64_t umax64(uint64_t x, uint64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t shl8(uint8_t x, uint8_t y)
{
    return x << y;
}
static inline uint16_t shl16(uint16_t x, uint16_t y)
{
    return x << y;
}
static inline uint32_t shl32(uint32_t x, uint32_t y)
{
    return x << y;
}
static inline uint64_t shl64(uint64_t x, uint64_t y)
{
    return x << y;
}
static inline uint8_t lshr8(uint8_t x, uint8_t y)
{
    return x >> y;
}
static inline uint16_t lshr16(uint16_t x, uint16_t y)
{
    return x >> y;
}
static inline uint32_t lshr32(uint32_t x, uint32_t y)
{
    return x >> y;
}
static inline uint64_t lshr64(uint64_t x, uint64_t y)
{
    return x >> y;
}
static inline int8_t ashr8(int8_t x, int8_t y)
{
    return x >> y;
}
static inline int16_t ashr16(int16_t x, int16_t y)
{
    return x >> y;
}
static inline int32_t ashr32(int32_t x, int32_t y)
{
    return x >> y;
}
static inline int64_t ashr64(int64_t x, int64_t y)
{
    return x >> y;
}
static inline uint8_t and8(uint8_t x, uint8_t y)
{
    return x & y;
}
static inline uint16_t and16(uint16_t x, uint16_t y)
{
    return x & y;
}
static inline uint32_t and32(uint32_t x, uint32_t y)
{
    return x & y;
}
static inline uint64_t and64(uint64_t x, uint64_t y)
{
    return x & y;
}
static inline uint8_t or8(uint8_t x, uint8_t y)
{
    return x | y;
}
static inline uint16_t or16(uint16_t x, uint16_t y)
{
    return x | y;
}
static inline uint32_t or32(uint32_t x, uint32_t y)
{
    return x | y;
}
static inline uint64_t or64(uint64_t x, uint64_t y)
{
    return x | y;
}
static inline uint8_t xor8(uint8_t x, uint8_t y)
{
    return x ^ y;
}
static inline uint16_t xor16(uint16_t x, uint16_t y)
{
    return x ^ y;
}
static inline uint32_t xor32(uint32_t x, uint32_t y)
{
    return x ^ y;
}
static inline uint64_t xor64(uint64_t x, uint64_t y)
{
    return x ^ y;
}
static inline bool ult8(uint8_t x, uint8_t y)
{
    return x < y;
}
static inline bool ult16(uint16_t x, uint16_t y)
{
    return x < y;
}
static inline bool ult32(uint32_t x, uint32_t y)
{
    return x < y;
}
static inline bool ult64(uint64_t x, uint64_t y)
{
    return x < y;
}
static inline bool ule8(uint8_t x, uint8_t y)
{
    return x <= y;
}
static inline bool ule16(uint16_t x, uint16_t y)
{
    return x <= y;
}
static inline bool ule32(uint32_t x, uint32_t y)
{
    return x <= y;
}
static inline bool ule64(uint64_t x, uint64_t y)
{
    return x <= y;
}
static inline bool slt8(int8_t x, int8_t y)
{
    return x < y;
}
static inline bool slt16(int16_t x, int16_t y)
{
    return x < y;
}
static inline bool slt32(int32_t x, int32_t y)
{
    return x < y;
}
static inline bool slt64(int64_t x, int64_t y)
{
    return x < y;
}
static inline bool sle8(int8_t x, int8_t y)
{
    return x <= y;
}
static inline bool sle16(int16_t x, int16_t y)
{
    return x <= y;
}
static inline bool sle32(int32_t x, int32_t y)
{
    return x <= y;
}
static inline bool sle64(int64_t x, int64_t y)
{
    return x <= y;
}
static inline int8_t pow8(int8_t x, int8_t y)
{
    int8_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int16_t pow16(int16_t x, int16_t y)
{
    int16_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int32_t pow32(int32_t x, int32_t y)
{
    int32_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int64_t pow64(int64_t x, int64_t y)
{
    int64_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline bool itob_i8_bool(int8_t x)
{
    return x;
}
static inline bool itob_i16_bool(int16_t x)
{
    return x;
}
static inline bool itob_i32_bool(int32_t x)
{
    return x;
}
static inline bool itob_i64_bool(int64_t x)
{
    return x;
}
static inline int8_t btoi_bool_i8(bool x)
{
    return x;
}
static inline int16_t btoi_bool_i16(bool x)
{
    return x;
}
static inline int32_t btoi_bool_i32(bool x)
{
    return x;
}
static inline int64_t btoi_bool_i64(bool x)
{
    return x;
}
#define sext_i8_i8(x) ((int8_t) (int8_t) x)
#define sext_i8_i16(x) ((int16_t) (int8_t) x)
#define sext_i8_i32(x) ((int32_t) (int8_t) x)
#define sext_i8_i64(x) ((int64_t) (int8_t) x)
#define sext_i16_i8(x) ((int8_t) (int16_t) x)
#define sext_i16_i16(x) ((int16_t) (int16_t) x)
#define sext_i16_i32(x) ((int32_t) (int16_t) x)
#define sext_i16_i64(x) ((int64_t) (int16_t) x)
#define sext_i32_i8(x) ((int8_t) (int32_t) x)
#define sext_i32_i16(x) ((int16_t) (int32_t) x)
#define sext_i32_i32(x) ((int32_t) (int32_t) x)
#define sext_i32_i64(x) ((int64_t) (int32_t) x)
#define sext_i64_i8(x) ((int8_t) (int64_t) x)
#define sext_i64_i16(x) ((int16_t) (int64_t) x)
#define sext_i64_i32(x) ((int32_t) (int64_t) x)
#define sext_i64_i64(x) ((int64_t) (int64_t) x)
#define zext_i8_i8(x) ((int8_t) (uint8_t) x)
#define zext_i8_i16(x) ((int16_t) (uint8_t) x)
#define zext_i8_i32(x) ((int32_t) (uint8_t) x)
#define zext_i8_i64(x) ((int64_t) (uint8_t) x)
#define zext_i16_i8(x) ((int8_t) (uint16_t) x)
#define zext_i16_i16(x) ((int16_t) (uint16_t) x)
#define zext_i16_i32(x) ((int32_t) (uint16_t) x)
#define zext_i16_i64(x) ((int64_t) (uint16_t) x)
#define zext_i32_i8(x) ((int8_t) (uint32_t) x)
#define zext_i32_i16(x) ((int16_t) (uint32_t) x)
#define zext_i32_i32(x) ((int32_t) (uint32_t) x)
#define zext_i32_i64(x) ((int64_t) (uint32_t) x)
#define zext_i64_i8(x) ((int8_t) (uint64_t) x)
#define zext_i64_i16(x) ((int16_t) (uint64_t) x)
#define zext_i64_i32(x) ((int32_t) (uint64_t) x)
#define zext_i64_i64(x) ((int64_t) (uint64_t) x)
#if defined(__OPENCL_VERSION__)
static int32_t futrts_popc8(int8_t x)
{
    return popcount(x);
}
static int32_t futrts_popc16(int16_t x)
{
    return popcount(x);
}
static int32_t futrts_popc32(int32_t x)
{
    return popcount(x);
}
static int32_t futrts_popc64(int64_t x)
{
    return popcount(x);
}
#elif defined(__CUDA_ARCH__)
static int32_t futrts_popc8(int8_t x)
{
    return __popc(zext_i8_i32(x));
}
static int32_t futrts_popc16(int16_t x)
{
    return __popc(zext_i16_i32(x));
}
static int32_t futrts_popc32(int32_t x)
{
    return __popc(x);
}
static int32_t futrts_popc64(int64_t x)
{
    return __popcll(x);
}
#else
static int32_t futrts_popc8(int8_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc16(int16_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc32(int32_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc64(int64_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
#endif
#if defined(__OPENCL_VERSION__)
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    return mul_hi(a, b);
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    return mul_hi(a, b);
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    return mul_hi(a, b);
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    return mul_hi(a, b);
}
#elif defined(__CUDA_ARCH__)
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    uint16_t aa = a;
    uint16_t bb = b;
    
    return aa * bb >> 8;
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    uint32_t aa = a;
    uint32_t bb = b;
    
    return aa * bb >> 16;
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    return mulhi(a, b);
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    return mul64hi(a, b);
}
#else
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    uint16_t aa = a;
    uint16_t bb = b;
    
    return aa * bb >> 8;
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    uint32_t aa = a;
    uint32_t bb = b;
    
    return aa * bb >> 16;
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    uint64_t aa = a;
    uint64_t bb = b;
    
    return aa * bb >> 32;
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    __uint128_t aa = a;
    __uint128_t bb = b;
    
    return aa * bb >> 64;
}
#endif
#if defined(__OPENCL_VERSION__)
static uint8_t futrts_mad_hi8(uint8_t a, uint8_t b, uint8_t c)
{
    return mad_hi(a, b, c);
}
static uint16_t futrts_mad_hi16(uint16_t a, uint16_t b, uint16_t c)
{
    return mad_hi(a, b, c);
}
static uint32_t futrts_mad_hi32(uint32_t a, uint32_t b, uint32_t c)
{
    return mad_hi(a, b, c);
}
static uint64_t futrts_mad_hi64(uint64_t a, uint64_t b, uint64_t c)
{
    return mad_hi(a, b, c);
}
#else
static uint8_t futrts_mad_hi8(uint8_t a, uint8_t b, uint8_t c)
{
    return futrts_mul_hi8(a, b) + c;
}
static uint16_t futrts_mad_hi16(uint16_t a, uint16_t b, uint16_t c)
{
    return futrts_mul_hi16(a, b) + c;
}
static uint32_t futrts_mad_hi32(uint32_t a, uint32_t b, uint32_t c)
{
    return futrts_mul_hi32(a, b) + c;
}
static uint64_t futrts_mad_hi64(uint64_t a, uint64_t b, uint64_t c)
{
    return futrts_mul_hi64(a, b) + c;
}
#endif
#if defined(__OPENCL_VERSION__)
static int32_t futrts_clzz8(int8_t x)
{
    return clz(x);
}
static int32_t futrts_clzz16(int16_t x)
{
    return clz(x);
}
static int32_t futrts_clzz32(int32_t x)
{
    return clz(x);
}
static int32_t futrts_clzz64(int64_t x)
{
    return clz(x);
}
#elif defined(__CUDA_ARCH__)
static int32_t futrts_clzz8(int8_t x)
{
    return __clz(zext_i8_i32(x)) - 24;
}
static int32_t futrts_clzz16(int16_t x)
{
    return __clz(zext_i16_i32(x)) - 16;
}
static int32_t futrts_clzz32(int32_t x)
{
    return __clz(x);
}
static int32_t futrts_clzz64(int64_t x)
{
    return __clzll(x);
}
#else
static int32_t futrts_clzz8(int8_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz16(int16_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz32(int32_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz64(int64_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
#endif
#if defined(__OPENCL_VERSION__)
static int32_t futrts_ctzz8(int8_t x)
{
    int i = 0;
    
    for (; i < 8 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
static int32_t futrts_ctzz16(int16_t x)
{
    int i = 0;
    
    for (; i < 16 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
static int32_t futrts_ctzz32(int32_t x)
{
    int i = 0;
    
    for (; i < 32 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
static int32_t futrts_ctzz64(int64_t x)
{
    int i = 0;
    
    for (; i < 64 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
#elif defined(__CUDA_ARCH__)
static int32_t futrts_ctzz8(int8_t x)
{
    int y = __ffs(x);
    
    return y == 0 ? 8 : y - 1;
}
static int32_t futrts_ctzz16(int16_t x)
{
    int y = __ffs(x);
    
    return y == 0 ? 16 : y - 1;
}
static int32_t futrts_ctzz32(int32_t x)
{
    int y = __ffs(x);
    
    return y == 0 ? 32 : y - 1;
}
static int32_t futrts_ctzz64(int64_t x)
{
    int y = __ffsll(x);
    
    return y == 0 ? 64 : y - 1;
}
#else
static int32_t futrts_ctzz8(int8_t x)
{
    return x == 0 ? 8 : __builtin_ctz((uint32_t) x);
}
static int32_t futrts_ctzz16(int16_t x)
{
    return x == 0 ? 16 : __builtin_ctz((uint32_t) x);
}
static int32_t futrts_ctzz32(int32_t x)
{
    return x == 0 ? 32 : __builtin_ctz(x);
}
static int32_t futrts_ctzz64(int64_t x)
{
    return x == 0 ? 64 : __builtin_ctzll(x);
}
#endif
static inline float fdiv32(float x, float y)
{
    return x / y;
}
static inline float fadd32(float x, float y)
{
    return x + y;
}
static inline float fsub32(float x, float y)
{
    return x - y;
}
static inline float fmul32(float x, float y)
{
    return x * y;
}
static inline float fmin32(float x, float y)
{
    return fmin(x, y);
}
static inline float fmax32(float x, float y)
{
    return fmax(x, y);
}
static inline float fpow32(float x, float y)
{
    return pow(x, y);
}
static inline bool cmplt32(float x, float y)
{
    return x < y;
}
static inline bool cmple32(float x, float y)
{
    return x <= y;
}
static inline float sitofp_i8_f32(int8_t x)
{
    return (float) x;
}
static inline float sitofp_i16_f32(int16_t x)
{
    return (float) x;
}
static inline float sitofp_i32_f32(int32_t x)
{
    return (float) x;
}
static inline float sitofp_i64_f32(int64_t x)
{
    return (float) x;
}
static inline float uitofp_i8_f32(uint8_t x)
{
    return (float) x;
}
static inline float uitofp_i16_f32(uint16_t x)
{
    return (float) x;
}
static inline float uitofp_i32_f32(uint32_t x)
{
    return (float) x;
}
static inline float uitofp_i64_f32(uint64_t x)
{
    return (float) x;
}
static inline int8_t fptosi_f32_i8(float x)
{
    return (int8_t) x;
}
static inline int16_t fptosi_f32_i16(float x)
{
    return (int16_t) x;
}
static inline int32_t fptosi_f32_i32(float x)
{
    return (int32_t) x;
}
static inline int64_t fptosi_f32_i64(float x)
{
    return (int64_t) x;
}
static inline uint8_t fptoui_f32_i8(float x)
{
    return (uint8_t) x;
}
static inline uint16_t fptoui_f32_i16(float x)
{
    return (uint16_t) x;
}
static inline uint32_t fptoui_f32_i32(float x)
{
    return (uint32_t) x;
}
static inline uint64_t fptoui_f32_i64(float x)
{
    return (uint64_t) x;
}
static inline double fdiv64(double x, double y)
{
    return x / y;
}
static inline double fadd64(double x, double y)
{
    return x + y;
}
static inline double fsub64(double x, double y)
{
    return x - y;
}
static inline double fmul64(double x, double y)
{
    return x * y;
}
static inline double fmin64(double x, double y)
{
    return fmin(x, y);
}
static inline double fmax64(double x, double y)
{
    return fmax(x, y);
}
static inline double fpow64(double x, double y)
{
    return pow(x, y);
}
static inline bool cmplt64(double x, double y)
{
    return x < y;
}
static inline bool cmple64(double x, double y)
{
    return x <= y;
}
static inline double sitofp_i8_f64(int8_t x)
{
    return (double) x;
}
static inline double sitofp_i16_f64(int16_t x)
{
    return (double) x;
}
static inline double sitofp_i32_f64(int32_t x)
{
    return (double) x;
}
static inline double sitofp_i64_f64(int64_t x)
{
    return (double) x;
}
static inline double uitofp_i8_f64(uint8_t x)
{
    return (double) x;
}
static inline double uitofp_i16_f64(uint16_t x)
{
    return (double) x;
}
static inline double uitofp_i32_f64(uint32_t x)
{
    return (double) x;
}
static inline double uitofp_i64_f64(uint64_t x)
{
    return (double) x;
}
static inline int8_t fptosi_f64_i8(double x)
{
    return (int8_t) x;
}
static inline int16_t fptosi_f64_i16(double x)
{
    return (int16_t) x;
}
static inline int32_t fptosi_f64_i32(double x)
{
    return (int32_t) x;
}
static inline int64_t fptosi_f64_i64(double x)
{
    return (int64_t) x;
}
static inline uint8_t fptoui_f64_i8(double x)
{
    return (uint8_t) x;
}
static inline uint16_t fptoui_f64_i16(double x)
{
    return (uint16_t) x;
}
static inline uint32_t fptoui_f64_i32(double x)
{
    return (uint32_t) x;
}
static inline uint64_t fptoui_f64_i64(double x)
{
    return (uint64_t) x;
}
static inline float fpconv_f32_f32(float x)
{
    return (float) x;
}
static inline double fpconv_f32_f64(float x)
{
    return (double) x;
}
static inline float fpconv_f64_f32(double x)
{
    return (float) x;
}
static inline double fpconv_f64_f64(double x)
{
    return (double) x;
}
static inline float futrts_log32(float x)
{
    return log(x);
}
static inline float futrts_log2_32(float x)
{
    return log2(x);
}
static inline float futrts_log10_32(float x)
{
    return log10(x);
}
static inline float futrts_sqrt32(float x)
{
    return sqrt(x);
}
static inline float futrts_exp32(float x)
{
    return exp(x);
}
static inline float futrts_cos32(float x)
{
    return cos(x);
}
static inline float futrts_sin32(float x)
{
    return sin(x);
}
static inline float futrts_tan32(float x)
{
    return tan(x);
}
static inline float futrts_acos32(float x)
{
    return acos(x);
}
static inline float futrts_asin32(float x)
{
    return asin(x);
}
static inline float futrts_atan32(float x)
{
    return atan(x);
}
static inline float futrts_cosh32(float x)
{
    return cosh(x);
}
static inline float futrts_sinh32(float x)
{
    return sinh(x);
}
static inline float futrts_tanh32(float x)
{
    return tanh(x);
}
static inline float futrts_acosh32(float x)
{
    return acosh(x);
}
static inline float futrts_asinh32(float x)
{
    return asinh(x);
}
static inline float futrts_atanh32(float x)
{
    return atanh(x);
}
static inline float futrts_atan2_32(float x, float y)
{
    return atan2(x, y);
}
static inline float futrts_gamma32(float x)
{
    return tgamma(x);
}
static inline float futrts_lgamma32(float x)
{
    return lgamma(x);
}
static inline bool futrts_isnan32(float x)
{
    return isnan(x);
}
static inline bool futrts_isinf32(float x)
{
    return isinf(x);
}
static inline int32_t futrts_to_bits32(float x)
{
    union {
        float f;
        int32_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline float futrts_from_bits32(int32_t x)
{
    union {
        int32_t f;
        float t;
    } p;
    
    p.f = x;
    return p.t;
}
#ifdef __OPENCL_VERSION__
static inline float fmod32(float x, float y)
{
    return fmod(x, y);
}
static inline float futrts_round32(float x)
{
    return rint(x);
}
static inline float futrts_floor32(float x)
{
    return floor(x);
}
static inline float futrts_ceil32(float x)
{
    return ceil(x);
}
static inline float futrts_lerp32(float v0, float v1, float t)
{
    return mix(v0, v1, t);
}
static inline float futrts_mad32(float a, float b, float c)
{
    return mad(a, b, c);
}
static inline float futrts_fma32(float a, float b, float c)
{
    return fma(a, b, c);
}
#else
static inline float fmod32(float x, float y)
{
    return fmodf(x, y);
}
static inline float futrts_round32(float x)
{
    return rintf(x);
}
static inline float futrts_floor32(float x)
{
    return floorf(x);
}
static inline float futrts_ceil32(float x)
{
    return ceilf(x);
}
static inline float futrts_lerp32(float v0, float v1, float t)
{
    return v0 + (v1 - v0) * t;
}
static inline float futrts_mad32(float a, float b, float c)
{
    return a * b + c;
}
static inline float futrts_fma32(float a, float b, float c)
{
    return fmaf(a, b, c);
}
#endif
static inline double futrts_log64(double x)
{
    return log(x);
}
static inline double futrts_log2_64(double x)
{
    return log2(x);
}
static inline double futrts_log10_64(double x)
{
    return log10(x);
}
static inline double futrts_sqrt64(double x)
{
    return sqrt(x);
}
static inline double futrts_exp64(double x)
{
    return exp(x);
}
static inline double futrts_cos64(double x)
{
    return cos(x);
}
static inline double futrts_sin64(double x)
{
    return sin(x);
}
static inline double futrts_tan64(double x)
{
    return tan(x);
}
static inline double futrts_acos64(double x)
{
    return acos(x);
}
static inline double futrts_asin64(double x)
{
    return asin(x);
}
static inline double futrts_atan64(double x)
{
    return atan(x);
}
static inline double futrts_cosh64(double x)
{
    return cosh(x);
}
static inline double futrts_sinh64(double x)
{
    return sinh(x);
}
static inline double futrts_tanh64(double x)
{
    return tanh(x);
}
static inline double futrts_acosh64(double x)
{
    return acosh(x);
}
static inline double futrts_asinh64(double x)
{
    return asinh(x);
}
static inline double futrts_atanh64(double x)
{
    return atanh(x);
}
static inline double futrts_atan2_64(double x, double y)
{
    return atan2(x, y);
}
static inline double futrts_gamma64(double x)
{
    return tgamma(x);
}
static inline double futrts_lgamma64(double x)
{
    return lgamma(x);
}
static inline double futrts_fma64(double a, double b, double c)
{
    return fma(a, b, c);
}
static inline double futrts_round64(double x)
{
    return rint(x);
}
static inline double futrts_ceil64(double x)
{
    return ceil(x);
}
static inline double futrts_floor64(double x)
{
    return floor(x);
}
static inline bool futrts_isnan64(double x)
{
    return isnan(x);
}
static inline bool futrts_isinf64(double x)
{
    return isinf(x);
}
static inline int64_t futrts_to_bits64(double x)
{
    union {
        double f;
        int64_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double futrts_from_bits64(int64_t x)
{
    union {
        int64_t f;
        double t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double fmod64(double x, double y)
{
    return fmod(x, y);
}
#ifdef __OPENCL_VERSION__
static inline double futrts_lerp64(double v0, double v1, double t)
{
    return mix(v0, v1, t);
}
static inline double futrts_mad64(double a, double b, double c)
{
    return mad(a, b, c);
}
#else
static inline double futrts_lerp64(double v0, double v1, double t)
{
    return v0 + (v1 - v0) * t;
}
static inline double futrts_mad64(double a, double b, double c)
{
    return a * b + c;
}
#endif
static int init_constants(struct futhark_context *);
static int free_constants(struct futhark_context *);
struct memblock_device {
    int *references;
    cl_mem mem;
    int64_t size;
    const char *desc;
} ;
struct memblock {
    int *references;
    char *mem;
    int64_t size;
    const char *desc;
} ;
typedef cl_mem fl_mem_t;
// Start of free_list.h.

// An entry in the free list.  May be invalid, to avoid having to
// deallocate entries as soon as they are removed.  There is also a
// tag, to help with memory reuse.
struct free_list_entry {
  size_t size;
  fl_mem_t mem;
  const char *tag;
  unsigned char valid;
};

struct free_list {
  struct free_list_entry *entries;        // Pointer to entries.
  int capacity;                           // Number of entries.
  int used;                               // Number of valid entries.
};

static void free_list_init(struct free_list *l) {
  l->capacity = 30; // Picked arbitrarily.
  l->used = 0;
  l->entries = (struct free_list_entry*) malloc(sizeof(struct free_list_entry) * l->capacity);
  for (int i = 0; i < l->capacity; i++) {
    l->entries[i].valid = 0;
  }
}

// Remove invalid entries from the free list.
static void free_list_pack(struct free_list *l) {
  int p = 0;
  for (int i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid) {
      l->entries[p] = l->entries[i];
      p++;
    }
  }

  // Now p is the number of used elements.  We don't want it to go
  // less than the default capacity (although in practice it's OK as
  // long as it doesn't become 1).
  if (p < 30) {
    p = 30;
  }
  l->entries = realloc(l->entries, p * sizeof(struct free_list_entry));
  l->capacity = p;
}

static void free_list_destroy(struct free_list *l) {
  assert(l->used == 0);
  free(l->entries);
}

static int free_list_find_invalid(struct free_list *l) {
  int i;
  for (i = 0; i < l->capacity; i++) {
    if (!l->entries[i].valid) {
      break;
    }
  }
  return i;
}

static void free_list_insert(struct free_list *l, size_t size, fl_mem_t mem, const char *tag) {
  int i = free_list_find_invalid(l);

  if (i == l->capacity) {
    // List is full; so we have to grow it.
    int new_capacity = l->capacity * 2 * sizeof(struct free_list_entry);
    l->entries = realloc(l->entries, new_capacity);
    for (int j = 0; j < l->capacity; j++) {
      l->entries[j+l->capacity].valid = 0;
    }
    l->capacity *= 2;
  }

  // Now 'i' points to the first invalid entry.
  l->entries[i].valid = 1;
  l->entries[i].size = size;
  l->entries[i].mem = mem;
  l->entries[i].tag = tag;

  l->used++;
}

// Find and remove a memory block of the indicated tag, or if that
// does not exist, another memory block with exactly the desired size.
// Returns 0 on success.
static int free_list_find(struct free_list *l, size_t size,
                          size_t *size_out, fl_mem_t *mem_out) {
  int size_match = -1;
  int i;
  for (i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid &&
        size <= l->entries[i].size &&
        (size_match < 0 || l->entries[i].size < l->entries[size_match].size)) {
      // If this entry is valid, has sufficient size, and is smaller than the
      // best entry found so far, use this entry.
      size_match = i;
    }
  }

  if (size_match >= 0) {
    l->entries[size_match].valid = 0;
    *size_out = l->entries[size_match].size;
    *mem_out = l->entries[size_match].mem;
    l->used--;
    return 0;
  } else {
    return 1;
  }
}

// Remove the first block in the free list.  Returns 0 if a block was
// removed, and nonzero if the free list was already empty.
static int free_list_first(struct free_list *l, fl_mem_t *mem_out) {
  for (int i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid) {
      l->entries[i].valid = 0;
      *mem_out = l->entries[i].mem;
      l->used--;
      return 0;
    }
  }

  return 1;
}

// End of free_list.h.

// Start of opencl.h.

#define OPENCL_SUCCEED_FATAL(e) opencl_succeed_fatal(e, #e, __FILE__, __LINE__)
#define OPENCL_SUCCEED_NONFATAL(e) opencl_succeed_nonfatal(e, #e, __FILE__, __LINE__)
// Take care not to override an existing error.
#define OPENCL_SUCCEED_OR_RETURN(e) {             \
    char *serror = OPENCL_SUCCEED_NONFATAL(e);    \
    if (serror) {                                 \
      if (!ctx->error) {                          \
        ctx->error = serror;                      \
        return bad;                               \
      } else {                                    \
        free(serror);                             \
      }                                           \
    }                                             \
  }

// OPENCL_SUCCEED_OR_RETURN returns the value of the variable 'bad' in
// scope.  By default, it will be this one.  Create a local variable
// of some other type if needed.  This is a bit of a hack, but it
// saves effort in the code generator.
static const int bad = 1;

struct opencl_config {
  int debugging;
  int profiling;
  int logging;
  int preferred_device_num;
  const char *preferred_platform;
  const char *preferred_device;
  int ignore_blacklist;

  const char* dump_program_to;
  const char* load_program_from;
  const char* dump_binary_to;
  const char* load_binary_from;

  size_t default_group_size;
  size_t default_num_groups;
  size_t default_tile_size;
  size_t default_threshold;

  int default_group_size_changed;
  int default_tile_size_changed;

  int num_sizes;
  const char **size_names;
  const char **size_vars;
  int64_t *size_values;
  const char **size_classes;
};

static void opencl_config_init(struct opencl_config *cfg,
                               int num_sizes,
                               const char *size_names[],
                               const char *size_vars[],
                               int64_t *size_values,
                               const char *size_classes[]) {
  cfg->debugging = 0;
  cfg->logging = 0;
  cfg->profiling = 0;
  cfg->preferred_device_num = 0;
  cfg->preferred_platform = "";
  cfg->preferred_device = "";
  cfg->ignore_blacklist = 0;
  cfg->dump_program_to = NULL;
  cfg->load_program_from = NULL;
  cfg->dump_binary_to = NULL;
  cfg->load_binary_from = NULL;

  // The following are dummy sizes that mean the concrete defaults
  // will be set during initialisation via hardware-inspection-based
  // heuristics.
  cfg->default_group_size = 0;
  cfg->default_num_groups = 0;
  cfg->default_tile_size = 0;
  cfg->default_threshold = 0;

  cfg->default_group_size_changed = 0;
  cfg->default_tile_size_changed = 0;

  cfg->num_sizes = num_sizes;
  cfg->size_names = size_names;
  cfg->size_vars = size_vars;
  cfg->size_values = size_values;
  cfg->size_classes = size_classes;
}

// A record of something that happened.
struct profiling_record {
  cl_event *event;
  int *runs;
  int64_t *runtime;
};

struct opencl_context {
  cl_device_id device;
  cl_context ctx;
  cl_command_queue queue;

  struct opencl_config cfg;

  struct free_list free_list;

  size_t max_group_size;
  size_t max_num_groups;
  size_t max_tile_size;
  size_t max_threshold;
  size_t max_local_memory;

  size_t lockstep_width;

  struct profiling_record *profiling_records;
  int profiling_records_capacity;
  int profiling_records_used;
};

struct opencl_device_option {
  cl_platform_id platform;
  cl_device_id device;
  cl_device_type device_type;
  char *platform_name;
  char *device_name;
};

// This function must be defined by the user.  It is invoked by
// setup_opencl() after the platform and device has been found, but
// before the program is loaded.  Its intended use is to tune
// constants based on the selected platform and device.
static void post_opencl_setup(struct opencl_context*, struct opencl_device_option*);

static char *strclone(const char *str) {
  size_t size = strlen(str) + 1;
  char *copy = (char*) malloc(size);
  if (copy == NULL) {
    return NULL;
  }

  memcpy(copy, str, size);
  return copy;
}

static const char* opencl_error_string(cl_int err)
{
    switch (err) {
        case CL_SUCCESS:                            return "Success!";
        case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                   return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
        case CL_MAP_FAILURE:                        return "Map failure";
        case CL_INVALID_VALUE:                      return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
        case CL_INVALID_PLATFORM:                   return "Invalid platform";
        case CL_INVALID_DEVICE:                     return "Invalid device";
        case CL_INVALID_CONTEXT:                    return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
        case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
        case CL_INVALID_SAMPLER:                    return "Invalid sampler";
        case CL_INVALID_BINARY:                     return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
        case CL_INVALID_PROGRAM:                    return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                     return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
        case CL_INVALID_EVENT:                      return "Invalid event";
        case CL_INVALID_OPERATION:                  return "Invalid operation";
        case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
        default:                                    return "Unknown";
    }
}

static void opencl_succeed_fatal(unsigned int ret,
                                 const char *call,
                                 const char *file,
                                 int line) {
  if (ret != CL_SUCCESS) {
    futhark_panic(-1, "%s:%d: OpenCL call\n  %s\nfailed with error code %d (%s)\n",
          file, line, call, ret, opencl_error_string(ret));
  }
}

static char* opencl_succeed_nonfatal(unsigned int ret,
                                     const char *call,
                                     const char *file,
                                     int line) {
  if (ret != CL_SUCCESS) {
    return msgprintf("%s:%d: OpenCL call\n  %s\nfailed with error code %d (%s)\n",
                     file, line, call, ret, opencl_error_string(ret));
  } else {
    return NULL;
  }
}

static void set_preferred_platform(struct opencl_config *cfg, const char *s) {
  cfg->preferred_platform = s;
  cfg->ignore_blacklist = 1;
}

static void set_preferred_device(struct opencl_config *cfg, const char *s) {
  int x = 0;
  if (*s == '#') {
    s++;
    while (isdigit(*s)) {
      x = x * 10 + (*s++)-'0';
    }
    // Skip trailing spaces.
    while (isspace(*s)) {
      s++;
    }
  }
  cfg->preferred_device = s;
  cfg->preferred_device_num = x;
  cfg->ignore_blacklist = 1;
}

static char* opencl_platform_info(cl_platform_id platform,
                                  cl_platform_info param) {
  size_t req_bytes;
  char *info;

  OPENCL_SUCCEED_FATAL(clGetPlatformInfo(platform, param, 0, NULL, &req_bytes));

  info = (char*) malloc(req_bytes);

  OPENCL_SUCCEED_FATAL(clGetPlatformInfo(platform, param, req_bytes, info, NULL));

  return info;
}

static char* opencl_device_info(cl_device_id device,
                                cl_device_info param) {
  size_t req_bytes;
  char *info;

  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device, param, 0, NULL, &req_bytes));

  info = (char*) malloc(req_bytes);

  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device, param, req_bytes, info, NULL));

  return info;
}

static void opencl_all_device_options(struct opencl_device_option **devices_out,
                                      size_t *num_devices_out) {
  size_t num_devices = 0, num_devices_added = 0;

  cl_platform_id *all_platforms;
  cl_uint *platform_num_devices;

  cl_uint num_platforms;

  // Find the number of platforms.
  OPENCL_SUCCEED_FATAL(clGetPlatformIDs(0, NULL, &num_platforms));

  // Make room for them.
  all_platforms = calloc(num_platforms, sizeof(cl_platform_id));
  platform_num_devices = calloc(num_platforms, sizeof(cl_uint));

  // Fetch all the platforms.
  OPENCL_SUCCEED_FATAL(clGetPlatformIDs(num_platforms, all_platforms, NULL));

  // Count the number of devices for each platform, as well as the
  // total number of devices.
  for (cl_uint i = 0; i < num_platforms; i++) {
    if (clGetDeviceIDs(all_platforms[i], CL_DEVICE_TYPE_ALL,
                       0, NULL, &platform_num_devices[i]) == CL_SUCCESS) {
      num_devices += platform_num_devices[i];
    } else {
      platform_num_devices[i] = 0;
    }
  }

  // Make room for all the device options.
  struct opencl_device_option *devices =
    calloc(num_devices, sizeof(struct opencl_device_option));

  // Loop through the platforms, getting information about their devices.
  for (cl_uint i = 0; i < num_platforms; i++) {
    cl_platform_id platform = all_platforms[i];
    cl_uint num_platform_devices = platform_num_devices[i];

    if (num_platform_devices == 0) {
      continue;
    }

    char *platform_name = opencl_platform_info(platform, CL_PLATFORM_NAME);
    cl_device_id *platform_devices =
      calloc(num_platform_devices, sizeof(cl_device_id));

    // Fetch all the devices.
    OPENCL_SUCCEED_FATAL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
                                  num_platform_devices, platform_devices, NULL));

    // Loop through the devices, adding them to the devices array.
    for (cl_uint i = 0; i < num_platform_devices; i++) {
      char *device_name = opencl_device_info(platform_devices[i], CL_DEVICE_NAME);
      devices[num_devices_added].platform = platform;
      devices[num_devices_added].device = platform_devices[i];
      OPENCL_SUCCEED_FATAL(clGetDeviceInfo(platform_devices[i], CL_DEVICE_TYPE,
                                     sizeof(cl_device_type),
                                     &devices[num_devices_added].device_type,
                                     NULL));
      // We don't want the structs to share memory, so copy the platform name.
      // Each device name is already unique.
      devices[num_devices_added].platform_name = strclone(platform_name);
      devices[num_devices_added].device_name = device_name;
      num_devices_added++;
    }
    free(platform_devices);
    free(platform_name);
  }
  free(all_platforms);
  free(platform_num_devices);

  *devices_out = devices;
  *num_devices_out = num_devices;
}

// Returns 0 on success.
static int list_devices(void) {
  struct opencl_device_option *devices;
  size_t num_devices;

  opencl_all_device_options(&devices, &num_devices);

  const char *cur_platform = "";
  for (size_t i = 0; i < num_devices; i++) {
    struct opencl_device_option device = devices[i];
    if (strcmp(cur_platform, device.platform_name) != 0) {
      printf("Platform: %s\n", device.platform_name);
      cur_platform = device.platform_name;
    }
    printf("[%d]: %s\n", (int)i, device.device_name);
  }

  // Free all the platform and device names.
  for (size_t j = 0; j < num_devices; j++) {
    free(devices[j].platform_name);
    free(devices[j].device_name);
  }
  free(devices);

  return 0;
}

// Returns 0 on success.
static int select_device_interactively(struct opencl_config *cfg) {
  struct opencl_device_option *devices;
  size_t num_devices;
  int ret = 1;

  opencl_all_device_options(&devices, &num_devices);

  printf("Choose OpenCL device:\n");
  const char *cur_platform = "";
  for (size_t i = 0; i < num_devices; i++) {
    struct opencl_device_option device = devices[i];
    if (strcmp(cur_platform, device.platform_name) != 0) {
      printf("Platform: %s\n", device.platform_name);
      cur_platform = device.platform_name;
    }
    printf("[%d] %s\n", (int)i, device.device_name);
  }

  int selection;
  printf("Choice: ");
  if (scanf("%d", &selection) == 1) {
    ret = 0;
    cfg->preferred_platform = "";
    cfg->preferred_device = "";
    cfg->preferred_device_num = selection;
    cfg->ignore_blacklist = 1;
  }

  // Free all the platform and device names.
  for (size_t j = 0; j < num_devices; j++) {
    free(devices[j].platform_name);
    free(devices[j].device_name);
  }
  free(devices);

  return ret;
}

static int is_blacklisted(const char *platform_name, const char *device_name,
                          const struct opencl_config *cfg) {
  if (strcmp(cfg->preferred_platform, "") != 0 ||
      strcmp(cfg->preferred_device, "") != 0) {
    return 0;
  } else if (strstr(platform_name, "Apple") != NULL &&
             strstr(device_name, "Intel(R) Core(TM)") != NULL) {
    return 1;
  } else {
    return 0;
  }
}

static struct opencl_device_option get_preferred_device(const struct opencl_config *cfg) {
  struct opencl_device_option *devices;
  size_t num_devices;

  opencl_all_device_options(&devices, &num_devices);

  int num_device_matches = 0;

  for (size_t i = 0; i < num_devices; i++) {
    struct opencl_device_option device = devices[i];
    if (strstr(device.platform_name, cfg->preferred_platform) != NULL &&
        strstr(device.device_name, cfg->preferred_device) != NULL &&
        (cfg->ignore_blacklist ||
         !is_blacklisted(device.platform_name, device.device_name, cfg)) &&
        num_device_matches++ == cfg->preferred_device_num) {
      // Free all the platform and device names, except the ones we have chosen.
      for (size_t j = 0; j < num_devices; j++) {
        if (j != i) {
          free(devices[j].platform_name);
          free(devices[j].device_name);
        }
      }
      free(devices);
      return device;
    }
  }

  futhark_panic(1, "Could not find acceptable OpenCL device.\n");
  exit(1); // Never reached
}

static void describe_device_option(struct opencl_device_option device) {
  fprintf(stderr, "Using platform: %s\n", device.platform_name);
  fprintf(stderr, "Using device: %s\n", device.device_name);
}

static cl_build_status build_opencl_program(cl_program program, cl_device_id device, const char* options) {
  cl_int clBuildProgram_error = clBuildProgram(program, 1, &device, options, NULL, NULL);

  // Avoid termination due to CL_BUILD_PROGRAM_FAILURE
  if (clBuildProgram_error != CL_SUCCESS &&
      clBuildProgram_error != CL_BUILD_PROGRAM_FAILURE) {
    OPENCL_SUCCEED_FATAL(clBuildProgram_error);
  }

  cl_build_status build_status;
  OPENCL_SUCCEED_FATAL(clGetProgramBuildInfo(program,
                                             device,
                                             CL_PROGRAM_BUILD_STATUS,
                                             sizeof(cl_build_status),
                                             &build_status,
                                             NULL));

  if (build_status != CL_SUCCESS) {
    char *build_log;
    size_t ret_val_size;
    OPENCL_SUCCEED_FATAL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size));

    build_log = (char*) malloc(ret_val_size+1);
    OPENCL_SUCCEED_FATAL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL));

    // The spec technically does not say whether the build log is zero-terminated, so let's be careful.
    build_log[ret_val_size] = '\0';

    fprintf(stderr, "Build log:\n%s\n", build_log);

    free(build_log);
  }

  return build_status;
}

// Fields in a bitmask indicating which types we must be sure are
// available.
enum opencl_required_type { OPENCL_F64 = 1 };

// We take as input several strings representing the program, because
// C does not guarantee that the compiler supports particularly large
// literals.  Notably, Visual C has a limit of 2048 characters.  The
// array must be NULL-terminated.
static cl_program setup_opencl_with_command_queue(struct opencl_context *ctx,
                                                  cl_command_queue queue,
                                                  const char *srcs[],
                                                  int required_types,
                                                  const char *extra_build_opts[]) {
  int error;

  free_list_init(&ctx->free_list);
  ctx->queue = queue;

  OPENCL_SUCCEED_FATAL(clGetCommandQueueInfo(ctx->queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx->ctx, NULL));

  // Fill out the device info.  This is redundant work if we are
  // called from setup_opencl() (which is the common case), but I
  // doubt it matters much.
  struct opencl_device_option device_option;
  OPENCL_SUCCEED_FATAL(clGetCommandQueueInfo(ctx->queue, CL_QUEUE_DEVICE,
                                       sizeof(cl_device_id),
                                       &device_option.device,
                                       NULL));
  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_PLATFORM,
                                 sizeof(cl_platform_id),
                                 &device_option.platform,
                                 NULL));
  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_TYPE,
                                 sizeof(cl_device_type),
                                 &device_option.device_type,
                                 NULL));
  device_option.platform_name = opencl_platform_info(device_option.platform, CL_PLATFORM_NAME);
  device_option.device_name = opencl_device_info(device_option.device, CL_DEVICE_NAME);

  ctx->device = device_option.device;

  if (required_types & OPENCL_F64) {
    cl_uint supported;
    OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
                                   sizeof(cl_uint), &supported, NULL));
    if (!supported) {
      futhark_panic(1, "Program uses double-precision floats, but this is not supported on the chosen device: %s\n",
            device_option.device_name);
    }
  }

  size_t max_group_size;
  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                 sizeof(size_t), &max_group_size, NULL));

  size_t max_tile_size = sqrt(max_group_size);

  cl_ulong max_local_memory;
  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_LOCAL_MEM_SIZE,
                                       sizeof(size_t), &max_local_memory, NULL));

  // Futhark reserves 4 bytes for bookkeeping information.
  max_local_memory -= 4;

  // The OpenCL implementation may reserve some local memory bytes for
  // various purposes.  In principle, we should use
  // clGetKernelWorkGroupInfo() to figure out for each kernel how much
  // is actually available, but our current code generator design
  // makes this infeasible.  Instead, we have this nasty hack where we
  // arbitrarily subtract some bytes, based on empirical measurements
  // (but which might be arbitrarily wrong).  Fortunately, we rarely
  // try to really push the local memory usage.
  if (strstr(device_option.platform_name, "NVIDIA CUDA") != NULL) {
    max_local_memory -= 12;
  } else if (strstr(device_option.platform_name, "AMD") != NULL) {
    max_local_memory -= 16;
  }

  // Make sure this function is defined.
  post_opencl_setup(ctx, &device_option);

  if (max_group_size < ctx->cfg.default_group_size) {
    if (ctx->cfg.default_group_size_changed) {
      fprintf(stderr, "Note: Device limits default group size to %zu (down from %zu).\n",
              max_group_size, ctx->cfg.default_group_size);
    }
    ctx->cfg.default_group_size = max_group_size;
  }

  if (max_tile_size < ctx->cfg.default_tile_size) {
    if (ctx->cfg.default_tile_size_changed) {
      fprintf(stderr, "Note: Device limits default tile size to %zu (down from %zu).\n",
              max_tile_size, ctx->cfg.default_tile_size);
    }
    ctx->cfg.default_tile_size = max_tile_size;
  }

  ctx->max_group_size = max_group_size;
  ctx->max_tile_size = max_tile_size; // No limit.
  ctx->max_threshold = ctx->max_num_groups = 0; // No limit.
  ctx->max_local_memory = max_local_memory;

  // Now we go through all the sizes, clamp them to the valid range,
  // or set them to the default.
  for (int i = 0; i < ctx->cfg.num_sizes; i++) {
    const char *size_class = ctx->cfg.size_classes[i];
    int64_t *size_value = &ctx->cfg.size_values[i];
    const char* size_name = ctx->cfg.size_names[i];
    int64_t max_value = 0, default_value = 0;

    if (strstr(size_class, "group_size") == size_class) {
      max_value = max_group_size;
      default_value = ctx->cfg.default_group_size;
    } else if (strstr(size_class, "num_groups") == size_class) {
      max_value = max_group_size; // Futhark assumes this constraint.
      default_value = ctx->cfg.default_num_groups;
      // XXX: as a quick and dirty hack, use twice as many threads for
      // histograms by default.  We really should just be smarter
      // about sizes somehow.
      if (strstr(size_name, ".seghist_") != NULL) {
        default_value *= 2;
      }
    } else if (strstr(size_class, "tile_size") == size_class) {
      max_value = sqrt(max_group_size);
      default_value = ctx->cfg.default_tile_size;
    } else if (strstr(size_class, "threshold") == size_class) {
      // Threshold can be as large as it takes.
      default_value = ctx->cfg.default_threshold;
    } else {
      // Bespoke sizes have no limit or default.
    }
    if (*size_value == 0) {
      *size_value = default_value;
    } else if (max_value > 0 && *size_value > max_value) {
      fprintf(stderr, "Note: Device limits %s to %d (down from %d)\n",
              size_name, (int)max_value, (int)*size_value);
      *size_value = max_value;
    }
  }

  if (ctx->lockstep_width == 0) {
    ctx->lockstep_width = 1;
  }

  if (ctx->cfg.logging) {
    fprintf(stderr, "Lockstep width: %d\n", (int)ctx->lockstep_width);
    fprintf(stderr, "Default group size: %d\n", (int)ctx->cfg.default_group_size);
    fprintf(stderr, "Default number of groups: %d\n", (int)ctx->cfg.default_num_groups);
  }

  char *fut_opencl_src = NULL;
  cl_program prog;
  error = CL_SUCCESS;

  if (ctx->cfg.load_binary_from == NULL) {
    size_t src_size = 0;

    // Maybe we have to read OpenCL source from somewhere else (used for debugging).
    if (ctx->cfg.load_program_from != NULL) {
      fut_opencl_src = slurp_file(ctx->cfg.load_program_from, NULL);
      assert(fut_opencl_src != NULL);
    } else {
      // Construct the OpenCL source concatenating all the fragments.
      for (const char **src = srcs; src && *src; src++) {
        src_size += strlen(*src);
      }

      fut_opencl_src = (char*) malloc(src_size + 1);

      size_t n, i;
      for (i = 0, n = 0; srcs && srcs[i]; i++) {
        strncpy(fut_opencl_src+n, srcs[i], src_size-n);
        n += strlen(srcs[i]);
      }
      fut_opencl_src[src_size] = 0;
    }

    if (ctx->cfg.dump_program_to != NULL) {
      if (ctx->cfg.debugging) {
        fprintf(stderr, "Dumping OpenCL source to %s...\n", ctx->cfg.dump_program_to);
      }

      dump_file(ctx->cfg.dump_program_to, fut_opencl_src, strlen(fut_opencl_src));
    }

    if (ctx->cfg.debugging) {
      fprintf(stderr, "Creating OpenCL program...\n");
    }

    const char* src_ptr[] = {fut_opencl_src};
    prog = clCreateProgramWithSource(ctx->ctx, 1, src_ptr, &src_size, &error);
    OPENCL_SUCCEED_FATAL(error);
  } else {
    if (ctx->cfg.debugging) {
      fprintf(stderr, "Loading OpenCL binary from %s...\n", ctx->cfg.load_binary_from);
    }
    size_t binary_size;
    unsigned char *fut_opencl_bin =
      (unsigned char*) slurp_file(ctx->cfg.load_binary_from, &binary_size);
    assert(fut_opencl_bin != NULL);
    const unsigned char *binaries[1] = { fut_opencl_bin };
    cl_int status = 0;

    prog = clCreateProgramWithBinary(ctx->ctx, 1, &device_option.device,
                                     &binary_size, binaries,
                                     &status, &error);

    OPENCL_SUCCEED_FATAL(status);
    OPENCL_SUCCEED_FATAL(error);
  }

  int compile_opts_size = 1024;

  for (int i = 0; i < ctx->cfg.num_sizes; i++) {
    compile_opts_size += strlen(ctx->cfg.size_names[i]) + 20;
  }

  for (int i = 0; extra_build_opts[i] != NULL; i++) {
    compile_opts_size += strlen(extra_build_opts[i] + 1);
  }

  char *compile_opts = (char*) malloc(compile_opts_size);

  int w = snprintf(compile_opts, compile_opts_size,
                   "-DLOCKSTEP_WIDTH=%d ",
                   (int)ctx->lockstep_width);

  for (int i = 0; i < ctx->cfg.num_sizes; i++) {
    w += snprintf(compile_opts+w, compile_opts_size-w,
                  "-D%s=%d ",
                  ctx->cfg.size_vars[i],
                  (int)ctx->cfg.size_values[i]);
  }

  for (int i = 0; extra_build_opts[i] != NULL; i++) {
    w += snprintf(compile_opts+w, compile_opts_size-w,
                  "%s ", extra_build_opts[i]);
  }

  if (ctx->cfg.debugging) {
    fprintf(stderr, "OpenCL compiler options: %s\n", compile_opts);
    fprintf(stderr, "Building OpenCL program...\n");
  }
  OPENCL_SUCCEED_FATAL(build_opencl_program(prog, device_option.device, compile_opts));

  free(compile_opts);
  free(fut_opencl_src);

  if (ctx->cfg.dump_binary_to != NULL) {
    if (ctx->cfg.debugging) {
      fprintf(stderr, "Dumping OpenCL binary to %s...\n", ctx->cfg.dump_binary_to);
    }

    size_t binary_size;
    OPENCL_SUCCEED_FATAL(clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES,
                                          sizeof(size_t), &binary_size, NULL));
    unsigned char *binary = (unsigned char*) malloc(binary_size);
    unsigned char *binaries[1] = { binary };
    OPENCL_SUCCEED_FATAL(clGetProgramInfo(prog, CL_PROGRAM_BINARIES,
                                          sizeof(unsigned char*), binaries, NULL));

    dump_file(ctx->cfg.dump_binary_to, binary, binary_size);
  }

  return prog;
}

static cl_program setup_opencl(struct opencl_context *ctx,
                               const char *srcs[],
                               int required_types,
                               const char *extra_build_opts[]) {

  ctx->lockstep_width = 0; // Real value set later.

  struct opencl_device_option device_option = get_preferred_device(&ctx->cfg);

  if (ctx->cfg.logging) {
    describe_device_option(device_option);
  }

  // Note that NVIDIA's OpenCL requires the platform property
  cl_context_properties properties[] = {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties)device_option.platform,
    0
  };

  cl_int clCreateContext_error;
  ctx->ctx = clCreateContext(properties, 1, &device_option.device, NULL, NULL, &clCreateContext_error);
  OPENCL_SUCCEED_FATAL(clCreateContext_error);

  cl_int clCreateCommandQueue_error;
  cl_command_queue queue =
    clCreateCommandQueue(ctx->ctx,
                         device_option.device,
                         ctx->cfg.profiling ? CL_QUEUE_PROFILING_ENABLE : 0,
                         &clCreateCommandQueue_error);
  OPENCL_SUCCEED_FATAL(clCreateCommandQueue_error);

  return setup_opencl_with_command_queue(ctx, queue, srcs, required_types, extra_build_opts);
}

// Count up the runtime all the profiling_records that occured during execution.
// Also clears the buffer of profiling_records.
static cl_int opencl_tally_profiling_records(struct opencl_context *ctx) {
  cl_int err;
  for (int i = 0; i < ctx->profiling_records_used; i++) {
    struct profiling_record record = ctx->profiling_records[i];

    cl_ulong start_t, end_t;

    if ((err = clGetEventProfilingInfo(*record.event,
                                       CL_PROFILING_COMMAND_START,
                                       sizeof(start_t),
                                       &start_t,
                                       NULL)) != CL_SUCCESS) {
      return err;
    }

    if ((err = clGetEventProfilingInfo(*record.event,
                                       CL_PROFILING_COMMAND_END,
                                       sizeof(end_t),
                                       &end_t,
                                       NULL)) != CL_SUCCESS) {
      return err;
    }

    // OpenCL provides nanosecond resolution, but we want
    // microseconds.
    *record.runs += 1;
    *record.runtime += (end_t - start_t)/1000;

    if ((err = clReleaseEvent(*record.event)) != CL_SUCCESS) {
      return err;
    }
    free(record.event);
  }

  ctx->profiling_records_used = 0;

  return CL_SUCCESS;
}

// If profiling, produce an event associated with a profiling record.
static cl_event* opencl_get_event(struct opencl_context *ctx, int *runs, int64_t *runtime) {
    if (ctx->profiling_records_used == ctx->profiling_records_capacity) {
      ctx->profiling_records_capacity *= 2;
      ctx->profiling_records =
        realloc(ctx->profiling_records,
                ctx->profiling_records_capacity *
                sizeof(struct profiling_record));
    }
    cl_event *event = malloc(sizeof(cl_event));
    ctx->profiling_records[ctx->profiling_records_used].event = event;
    ctx->profiling_records[ctx->profiling_records_used].runs = runs;
    ctx->profiling_records[ctx->profiling_records_used].runtime = runtime;
    ctx->profiling_records_used++;
    return event;
}

// Allocate memory from driver. The problem is that OpenCL may perform
// lazy allocation, so we cannot know whether an allocation succeeded
// until the first time we try to use it.  Hence we immediately
// perform a write to see if the allocation succeeded.  This is slow,
// but the assumption is that this operation will be rare (most things
// will go through the free list).
static int opencl_alloc_actual(struct opencl_context *ctx, size_t size, cl_mem *mem_out) {
  int error;
  *mem_out = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE, size, NULL, &error);

  if (error != CL_SUCCESS) {
    return error;
  }

  int x = 2;
  error = clEnqueueWriteBuffer(ctx->queue, *mem_out, 1, 0, sizeof(x), &x, 0, NULL, NULL);

  // No need to wait for completion here. clWaitForEvents() cannot
  // return mem object allocation failures. This implies that the
  // buffer is faulted onto the device on enqueue. (Observation by
  // Andreas Kloeckner.)

  return error;
}

static int opencl_alloc(struct opencl_context *ctx, size_t min_size, const char *tag, cl_mem *mem_out) {
  (void)tag;
  if (min_size < sizeof(int)) {
    min_size = sizeof(int);
  }

  size_t size;

  if (free_list_find(&ctx->free_list, min_size, &size, mem_out) == 0) {
    // Successfully found a free block.  Is it big enough?
    //
    // FIXME: we might also want to check whether the block is *too
    // big*, to avoid internal fragmentation.  However, this can
    // sharply impact performance on programs where arrays change size
    // frequently.  Fortunately, such allocations are usually fairly
    // short-lived, as they are necessarily within a loop, so the risk
    // of internal fragmentation resulting in an OOM situation is
    // limited.  However, it would be preferable if we could go back
    // and *shrink* oversize allocations when we encounter an OOM
    // condition.  That is technically feasible, since we do not
    // expose OpenCL pointer values directly to the application, but
    // instead rely on a level of indirection.
    if (size >= min_size) {
      if (ctx->cfg.debugging) {
        fprintf(stderr, "No need to allocate: Found a block in the free list.\n");
      }

      return CL_SUCCESS;
    } else {
      if (ctx->cfg.debugging) {
        fprintf(stderr, "Found a free block, but it was too small.\n");
      }

      // Not just right - free it.
      int error = clReleaseMemObject(*mem_out);
      if (error != CL_SUCCESS) {
        return error;
      }
    }
  }

  // We have to allocate a new block from the driver.  If the
  // allocation does not succeed, then we might be in an out-of-memory
  // situation.  We now start freeing things from the free list until
  // we think we have freed enough that the allocation will succeed.
  // Since we don't know how far the allocation is from fitting, we
  // have to check after every deallocation.  This might be pretty
  // expensive.  Let's hope that this case is hit rarely.

  if (ctx->cfg.debugging) {
    fprintf(stderr, "Actually allocating the desired block.\n");
  }

  int error = opencl_alloc_actual(ctx, min_size, mem_out);

  while (error == CL_MEM_OBJECT_ALLOCATION_FAILURE) {
    if (ctx->cfg.debugging) {
      fprintf(stderr, "Out of OpenCL memory: releasing entry from the free list...\n");
    }
    cl_mem mem;
    if (free_list_first(&ctx->free_list, &mem) == 0) {
      error = clReleaseMemObject(mem);
      if (error != CL_SUCCESS) {
        return error;
      }
    } else {
      break;
    }
    error = opencl_alloc_actual(ctx, min_size, mem_out);
  }

  return error;
}

static int opencl_free(struct opencl_context *ctx, cl_mem mem, const char *tag) {
  size_t size;
  cl_mem existing_mem;

  // If there is already a block with this tag, then remove it.
  if (free_list_find(&ctx->free_list, -1, &size, &existing_mem) == 0) {
    int error = clReleaseMemObject(existing_mem);
    if (error != CL_SUCCESS) {
      return error;
    }
  }

  int error = clGetMemObjectInfo(mem, CL_MEM_SIZE, sizeof(size_t), &size, NULL);

  if (error == CL_SUCCESS) {
    free_list_insert(&ctx->free_list, size, mem, tag);
  }

  return error;
}

static int opencl_free_all(struct opencl_context *ctx) {
  cl_mem mem;
  free_list_pack(&ctx->free_list);
  while (free_list_first(&ctx->free_list, &mem) == 0) {
    int error = clReleaseMemObject(mem);
    if (error != CL_SUCCESS) {
      return error;
    }
  }

  return CL_SUCCESS;
}

// Free everything that belongs to 'ctx', but do not free 'ctx'
// itself.
static void teardown_opencl(struct opencl_context *ctx) {
  (void)opencl_tally_profiling_records(ctx);
  free(ctx->profiling_records);
  (void)opencl_free_all(ctx);
  (void)clReleaseCommandQueue(ctx->queue);
  (void)clReleaseContext(ctx->ctx);
}

// End of opencl.h.

static const char *opencl_program[] =
                  {"#ifdef cl_clang_storage_class_specifiers\n#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable\n#endif\n#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable\n__kernel void dummy_kernel(__global unsigned char *dummy, int n)\n{\n    const int thread_gid = get_global_id(0);\n    \n    if (thread_gid >= n)\n        return;\n}\ntypedef char int8_t;\ntypedef short int16_t;\ntypedef int int32_t;\ntypedef long int64_t;\ntypedef uchar uint8_t;\ntypedef ushort uint16_t;\ntypedef uint uint32_t;\ntypedef ulong uint64_t;\n#ifdef cl_nv_pragma_unroll\nstatic inline void mem_fence_global()\n{\n    asm(\"membar.gl;\");\n}\n#else\nstatic inline void mem_fence_global()\n{\n    mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\n}\n#endif\nstatic inline void mem_fence_local()\n{\n    mem_fence(CLK_LOCAL_MEM_FENCE);\n}\nstatic inline uint8_t add8(uint8_t x, uint8_t y)\n{\n    return x + y;\n}\nstatic inline uint16_t add16(uint16_t x, uint16_t y)\n{\n    return x + y;\n}\nstatic inline uint32_t add32(uint32_t x, uint32_t y)\n{\n    return x + y;\n}\nstatic inline uint64_t add64(uint64_t x, uint64_t y)\n{\n    return x + y;\n}\nstatic inline uint8_t sub8(uint8_t x, uint8_t y)\n{\n    return x - y;\n}\nstatic inline uint16_t sub16(uint16_t x, uint16_t y)\n{\n    return x - y;\n}\nstatic inline uint32_t sub32(uint32_t x, uint32_t y)\n{\n    return x - y;\n}\nstatic inline uint64_t sub64(uint64_t x, uint64_t y)\n{\n    return x - y;\n}\nstatic inline uint8_t mul8(uint8_t x, uint8_t y)\n{\n    return x * y;\n}\nstatic inline uint16_t mul16(uint16_t x, uint16_t y)\n{\n    return x * y;\n}\nstatic inline uint32_t mul32(uint32_t x, uint32_t y)\n{\n    return x * y;\n}\nstatic inline uint64_t mul64(uint64_t x, uint64_t y)\n{\n    return x * y;\n}\nstatic inline uint8_t udiv8(uint8_t x, uint8_t y)\n{\n    return x / y;\n}\nstatic inline uint16_t udiv16(uint16_t x, uint16_t y)\n{\n    return x / y;\n}\nstatic inline uint32_t udiv32(uint32_t x, uint32_t y)\n{\n    return x / y;\n}\nstatic inline uint64_t udiv64(uint64_t x, uint64_t y)\n{\n    return x / y;\n}\nstatic ",
                   "inline uint8_t udiv_up8(uint8_t x, uint8_t y)\n{\n    return (x + y - 1) / y;\n}\nstatic inline uint16_t udiv_up16(uint16_t x, uint16_t y)\n{\n    return (x + y - 1) / y;\n}\nstatic inline uint32_t udiv_up32(uint32_t x, uint32_t y)\n{\n    return (x + y - 1) / y;\n}\nstatic inline uint64_t udiv_up64(uint64_t x, uint64_t y)\n{\n    return (x + y - 1) / y;\n}\nstatic inline uint8_t umod8(uint8_t x, uint8_t y)\n{\n    return x % y;\n}\nstatic inline uint16_t umod16(uint16_t x, uint16_t y)\n{\n    return x % y;\n}\nstatic inline uint32_t umod32(uint32_t x, uint32_t y)\n{\n    return x % y;\n}\nstatic inline uint64_t umod64(uint64_t x, uint64_t y)\n{\n    return x % y;\n}\nstatic inline uint8_t udiv_safe8(uint8_t x, uint8_t y)\n{\n    return y == 0 ? 0 : x / y;\n}\nstatic inline uint16_t udiv_safe16(uint16_t x, uint16_t y)\n{\n    return y == 0 ? 0 : x / y;\n}\nstatic inline uint32_t udiv_safe32(uint32_t x, uint32_t y)\n{\n    return y == 0 ? 0 : x / y;\n}\nstatic inline uint64_t udiv_safe64(uint64_t x, uint64_t y)\n{\n    return y == 0 ? 0 : x / y;\n}\nstatic inline uint8_t udiv_up_safe8(uint8_t x, uint8_t y)\n{\n    return y == 0 ? 0 : (x + y - 1) / y;\n}\nstatic inline uint16_t udiv_up_safe16(uint16_t x, uint16_t y)\n{\n    return y == 0 ? 0 : (x + y - 1) / y;\n}\nstatic inline uint32_t udiv_up_safe32(uint32_t x, uint32_t y)\n{\n    return y == 0 ? 0 : (x + y - 1) / y;\n}\nstatic inline uint64_t udiv_up_safe64(uint64_t x, uint64_t y)\n{\n    return y == 0 ? 0 : (x + y - 1) / y;\n}\nstatic inline uint8_t umod_safe8(uint8_t x, uint8_t y)\n{\n    return y == 0 ? 0 : x % y;\n}\nstatic inline uint16_t umod_safe16(uint16_t x, uint16_t y)\n{\n    return y == 0 ? 0 : x % y;\n}\nstatic inline uint32_t umod_safe32(uint32_t x, uint32_t y)\n{\n    return y == 0 ? 0 : x % y;\n}\nstatic inline uint64_t umod_safe64(uint64_t x, uint64_t y)\n{\n    return y == 0 ? 0 : x % y;\n}\nstatic inline int8_t sdiv8(int8_t x, int8_t y)\n{\n    int8_t q = x / y;\n    int8_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int16_t sdiv16(int16",
                   "_t x, int16_t y)\n{\n    int16_t q = x / y;\n    int16_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int32_t sdiv32(int32_t x, int32_t y)\n{\n    int32_t q = x / y;\n    int32_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int64_t sdiv64(int64_t x, int64_t y)\n{\n    int64_t q = x / y;\n    int64_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int8_t sdiv_up8(int8_t x, int8_t y)\n{\n    return sdiv8(x + y - 1, y);\n}\nstatic inline int16_t sdiv_up16(int16_t x, int16_t y)\n{\n    return sdiv16(x + y - 1, y);\n}\nstatic inline int32_t sdiv_up32(int32_t x, int32_t y)\n{\n    return sdiv32(x + y - 1, y);\n}\nstatic inline int64_t sdiv_up64(int64_t x, int64_t y)\n{\n    return sdiv64(x + y - 1, y);\n}\nstatic inline int8_t smod8(int8_t x, int8_t y)\n{\n    int8_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int16_t smod16(int16_t x, int16_t y)\n{\n    int16_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int32_t smod32(int32_t x, int32_t y)\n{\n    int32_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int64_t smod64(int64_t x, int64_t y)\n{\n    int64_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int8_t sdiv_safe8(int8_t x, int8_t y)\n{\n    return y == 0 ? 0 : sdiv8(x, y);\n}\nstatic inline int16_t sdiv_safe16(int16_t x, int16_t y)\n{\n    return y == 0 ? 0 : sdiv16(x, y);\n}\nstatic inline int32_t sdiv_safe32(int32_t x, int32_t y)\n{\n    return y == 0 ? 0 : sdiv32(x, y);\n}\nstatic inline int64_t sdiv_safe64(int64_t x, int64_t y)\n{\n    return y == 0 ? 0 : sdiv64(x, y);\n}\nstatic inline int8_t sdiv_up_safe8(int8_t x, int8_t y)\n{\n    return sdiv_safe8(x + y - 1, y);\n}\nstatic inline int16_t sdiv_up_safe16(int16_t x, int16_t y)\n{\n    return sdiv_safe16(x + y - 1, y",
                   ");\n}\nstatic inline int32_t sdiv_up_safe32(int32_t x, int32_t y)\n{\n    return sdiv_safe32(x + y - 1, y);\n}\nstatic inline int64_t sdiv_up_safe64(int64_t x, int64_t y)\n{\n    return sdiv_safe64(x + y - 1, y);\n}\nstatic inline int8_t smod_safe8(int8_t x, int8_t y)\n{\n    return y == 0 ? 0 : smod8(x, y);\n}\nstatic inline int16_t smod_safe16(int16_t x, int16_t y)\n{\n    return y == 0 ? 0 : smod16(x, y);\n}\nstatic inline int32_t smod_safe32(int32_t x, int32_t y)\n{\n    return y == 0 ? 0 : smod32(x, y);\n}\nstatic inline int64_t smod_safe64(int64_t x, int64_t y)\n{\n    return y == 0 ? 0 : smod64(x, y);\n}\nstatic inline int8_t squot8(int8_t x, int8_t y)\n{\n    return x / y;\n}\nstatic inline int16_t squot16(int16_t x, int16_t y)\n{\n    return x / y;\n}\nstatic inline int32_t squot32(int32_t x, int32_t y)\n{\n    return x / y;\n}\nstatic inline int64_t squot64(int64_t x, int64_t y)\n{\n    return x / y;\n}\nstatic inline int8_t srem8(int8_t x, int8_t y)\n{\n    return x % y;\n}\nstatic inline int16_t srem16(int16_t x, int16_t y)\n{\n    return x % y;\n}\nstatic inline int32_t srem32(int32_t x, int32_t y)\n{\n    return x % y;\n}\nstatic inline int64_t srem64(int64_t x, int64_t y)\n{\n    return x % y;\n}\nstatic inline int8_t squot_safe8(int8_t x, int8_t y)\n{\n    return y == 0 ? 0 : x / y;\n}\nstatic inline int16_t squot_safe16(int16_t x, int16_t y)\n{\n    return y == 0 ? 0 : x / y;\n}\nstatic inline int32_t squot_safe32(int32_t x, int32_t y)\n{\n    return y == 0 ? 0 : x / y;\n}\nstatic inline int64_t squot_safe64(int64_t x, int64_t y)\n{\n    return y == 0 ? 0 : x / y;\n}\nstatic inline int8_t srem_safe8(int8_t x, int8_t y)\n{\n    return y == 0 ? 0 : x % y;\n}\nstatic inline int16_t srem_safe16(int16_t x, int16_t y)\n{\n    return y == 0 ? 0 : x % y;\n}\nstatic inline int32_t srem_safe32(int32_t x, int32_t y)\n{\n    return y == 0 ? 0 : x % y;\n}\nstatic inline int64_t srem_safe64(int64_t x, int64_t y)\n{\n    return y == 0 ? 0 : x % y;\n}\nstatic inline int8_t smin8(int8_t x, int8_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int16_t smi",
                   "n16(int16_t x, int16_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int32_t smin32(int32_t x, int32_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int64_t smin64(int64_t x, int64_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint8_t umin8(uint8_t x, uint8_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint16_t umin16(uint16_t x, uint16_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint32_t umin32(uint32_t x, uint32_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint64_t umin64(uint64_t x, uint64_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int8_t smax8(int8_t x, int8_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int16_t smax16(int16_t x, int16_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int32_t smax32(int32_t x, int32_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int64_t smax64(int64_t x, int64_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint8_t umax8(uint8_t x, uint8_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint16_t umax16(uint16_t x, uint16_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint32_t umax32(uint32_t x, uint32_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint64_t umax64(uint64_t x, uint64_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint8_t shl8(uint8_t x, uint8_t y)\n{\n    return x << y;\n}\nstatic inline uint16_t shl16(uint16_t x, uint16_t y)\n{\n    return x << y;\n}\nstatic inline uint32_t shl32(uint32_t x, uint32_t y)\n{\n    return x << y;\n}\nstatic inline uint64_t shl64(uint64_t x, uint64_t y)\n{\n    return x << y;\n}\nstatic inline uint8_t lshr8(uint8_t x, uint8_t y)\n{\n    return x >> y;\n}\nstatic inline uint16_t lshr16(uint16_t x, uint16_t y)\n{\n    return x >> y;\n}\nstatic inline uint32_t lshr32(uint32_t x, uint32_t y)\n{\n    return x >> y;\n}\nstatic inline uint64_t lshr64(uint64_t x, uint64_t y)\n{\n    return x >> y;\n}\nstatic inline int8_t ashr8(int8_t x, int8_t y)\n{\n    return x >> y;\n}\nstatic inline int16_t ashr16(int16_t x, int16_t y)\n{\n    return x >> y;\n}\nstatic inline int32_t ashr32(int32_t x, int32_t y)",
                   "\n{\n    return x >> y;\n}\nstatic inline int64_t ashr64(int64_t x, int64_t y)\n{\n    return x >> y;\n}\nstatic inline uint8_t and8(uint8_t x, uint8_t y)\n{\n    return x & y;\n}\nstatic inline uint16_t and16(uint16_t x, uint16_t y)\n{\n    return x & y;\n}\nstatic inline uint32_t and32(uint32_t x, uint32_t y)\n{\n    return x & y;\n}\nstatic inline uint64_t and64(uint64_t x, uint64_t y)\n{\n    return x & y;\n}\nstatic inline uint8_t or8(uint8_t x, uint8_t y)\n{\n    return x | y;\n}\nstatic inline uint16_t or16(uint16_t x, uint16_t y)\n{\n    return x | y;\n}\nstatic inline uint32_t or32(uint32_t x, uint32_t y)\n{\n    return x | y;\n}\nstatic inline uint64_t or64(uint64_t x, uint64_t y)\n{\n    return x | y;\n}\nstatic inline uint8_t xor8(uint8_t x, uint8_t y)\n{\n    return x ^ y;\n}\nstatic inline uint16_t xor16(uint16_t x, uint16_t y)\n{\n    return x ^ y;\n}\nstatic inline uint32_t xor32(uint32_t x, uint32_t y)\n{\n    return x ^ y;\n}\nstatic inline uint64_t xor64(uint64_t x, uint64_t y)\n{\n    return x ^ y;\n}\nstatic inline bool ult8(uint8_t x, uint8_t y)\n{\n    return x < y;\n}\nstatic inline bool ult16(uint16_t x, uint16_t y)\n{\n    return x < y;\n}\nstatic inline bool ult32(uint32_t x, uint32_t y)\n{\n    return x < y;\n}\nstatic inline bool ult64(uint64_t x, uint64_t y)\n{\n    return x < y;\n}\nstatic inline bool ule8(uint8_t x, uint8_t y)\n{\n    return x <= y;\n}\nstatic inline bool ule16(uint16_t x, uint16_t y)\n{\n    return x <= y;\n}\nstatic inline bool ule32(uint32_t x, uint32_t y)\n{\n    return x <= y;\n}\nstatic inline bool ule64(uint64_t x, uint64_t y)\n{\n    return x <= y;\n}\nstatic inline bool slt8(int8_t x, int8_t y)\n{\n    return x < y;\n}\nstatic inline bool slt16(int16_t x, int16_t y)\n{\n    return x < y;\n}\nstatic inline bool slt32(int32_t x, int32_t y)\n{\n    return x < y;\n}\nstatic inline bool slt64(int64_t x, int64_t y)\n{\n    return x < y;\n}\nstatic inline bool sle8(int8_t x, int8_t y)\n{\n    return x <= y;\n}\nstatic inline bool sle16(int16_t x, int16_t y)\n{\n    return x <= y;\n}\nstatic inline bool sle32(int32_t x, int32_",
                   "t y)\n{\n    return x <= y;\n}\nstatic inline bool sle64(int64_t x, int64_t y)\n{\n    return x <= y;\n}\nstatic inline int8_t pow8(int8_t x, int8_t y)\n{\n    int8_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int16_t pow16(int16_t x, int16_t y)\n{\n    int16_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int32_t pow32(int32_t x, int32_t y)\n{\n    int32_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int64_t pow64(int64_t x, int64_t y)\n{\n    int64_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline bool itob_i8_bool(int8_t x)\n{\n    return x;\n}\nstatic inline bool itob_i16_bool(int16_t x)\n{\n    return x;\n}\nstatic inline bool itob_i32_bool(int32_t x)\n{\n    return x;\n}\nstatic inline bool itob_i64_bool(int64_t x)\n{\n    return x;\n}\nstatic inline int8_t btoi_bool_i8(bool x)\n{\n    return x;\n}\nstatic inline int16_t btoi_bool_i16(bool x)\n{\n    return x;\n}\nstatic inline int32_t btoi_bool_i32(bool x)\n{\n    return x;\n}\nstatic inline int64_t btoi_bool_i64(bool x)\n{\n    return x;\n}\n#define sext_i8_i8(x) ((int8_t) (int8_t) x)\n#define sext_i8_i16(x) ((int16_t) (int8_t) x)\n#define sext_i8_i32(x) ((int32_t) (int8_t) x)\n#define sext_i8_i64(x) ((int64_t) (int8_t) x)\n#define sext_i16_i8(x) ((int8_t) (int16_t) x)\n#define sext_i16_i16(x) ((int16_t) (int16_t) x)\n#define sext_i16_i32(x) ((int32_t) (int16_t) x)\n#define sext_i16_i64(x) ((int64_t) (int16_t) x)\n#define sext_i32_i8(x) ((int8_t) (int32_t) x)\n#define sext_i32_i16(x) ((int16_t) (int32_t) x)\n#define sext_i32_i32(x) ((int32_t) (int32_t) x)\n#define sext_i32_i64(x) ((int64_t) (int32_t) x)\n#defi",
                   "ne sext_i64_i8(x) ((int8_t) (int64_t) x)\n#define sext_i64_i16(x) ((int16_t) (int64_t) x)\n#define sext_i64_i32(x) ((int32_t) (int64_t) x)\n#define sext_i64_i64(x) ((int64_t) (int64_t) x)\n#define zext_i8_i8(x) ((int8_t) (uint8_t) x)\n#define zext_i8_i16(x) ((int16_t) (uint8_t) x)\n#define zext_i8_i32(x) ((int32_t) (uint8_t) x)\n#define zext_i8_i64(x) ((int64_t) (uint8_t) x)\n#define zext_i16_i8(x) ((int8_t) (uint16_t) x)\n#define zext_i16_i16(x) ((int16_t) (uint16_t) x)\n#define zext_i16_i32(x) ((int32_t) (uint16_t) x)\n#define zext_i16_i64(x) ((int64_t) (uint16_t) x)\n#define zext_i32_i8(x) ((int8_t) (uint32_t) x)\n#define zext_i32_i16(x) ((int16_t) (uint32_t) x)\n#define zext_i32_i32(x) ((int32_t) (uint32_t) x)\n#define zext_i32_i64(x) ((int64_t) (uint32_t) x)\n#define zext_i64_i8(x) ((int8_t) (uint64_t) x)\n#define zext_i64_i16(x) ((int16_t) (uint64_t) x)\n#define zext_i64_i32(x) ((int32_t) (uint64_t) x)\n#define zext_i64_i64(x) ((int64_t) (uint64_t) x)\n#if defined(__OPENCL_VERSION__)\nstatic int32_t futrts_popc8(int8_t x)\n{\n    return popcount(x);\n}\nstatic int32_t futrts_popc16(int16_t x)\n{\n    return popcount(x);\n}\nstatic int32_t futrts_popc32(int32_t x)\n{\n    return popcount(x);\n}\nstatic int32_t futrts_popc64(int64_t x)\n{\n    return popcount(x);\n}\n#elif defined(__CUDA_ARCH__)\nstatic int32_t futrts_popc8(int8_t x)\n{\n    return __popc(zext_i8_i32(x));\n}\nstatic int32_t futrts_popc16(int16_t x)\n{\n    return __popc(zext_i16_i32(x));\n}\nstatic int32_t futrts_popc32(int32_t x)\n{\n    return __popc(x);\n}\nstatic int32_t futrts_popc64(int64_t x)\n{\n    return __popcll(x);\n}\n#else\nstatic int32_t futrts_popc8(int8_t x)\n{\n    int c = 0;\n    \n    for (; x; ++c)\n        x &= x - 1;\n    return c;\n}\nstatic int32_t futrts_popc16(int16_t x)\n{\n    int c = 0;\n    \n    for (; x; ++c)\n        x &= x - 1;\n    return c;\n}\nstatic int32_t futrts_popc32(int32_t x)\n{\n    int c = 0;\n    \n    for (; x; ++c)\n        x &= x - 1;\n    return c;\n}\nstatic int32_t futrts_popc64(int64_t x)\n{\n    int c = 0;\n    \n    for ",
                   "(; x; ++c)\n        x &= x - 1;\n    return c;\n}\n#endif\n#if defined(__OPENCL_VERSION__)\nstatic uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)\n{\n    return mul_hi(a, b);\n}\nstatic uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)\n{\n    return mul_hi(a, b);\n}\nstatic uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)\n{\n    return mul_hi(a, b);\n}\nstatic uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)\n{\n    return mul_hi(a, b);\n}\n#elif defined(__CUDA_ARCH__)\nstatic uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)\n{\n    uint16_t aa = a;\n    uint16_t bb = b;\n    \n    return aa * bb >> 8;\n}\nstatic uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)\n{\n    uint32_t aa = a;\n    uint32_t bb = b;\n    \n    return aa * bb >> 16;\n}\nstatic uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)\n{\n    return mulhi(a, b);\n}\nstatic uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)\n{\n    return mul64hi(a, b);\n}\n#else\nstatic uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)\n{\n    uint16_t aa = a;\n    uint16_t bb = b;\n    \n    return aa * bb >> 8;\n}\nstatic uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)\n{\n    uint32_t aa = a;\n    uint32_t bb = b;\n    \n    return aa * bb >> 16;\n}\nstatic uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)\n{\n    uint64_t aa = a;\n    uint64_t bb = b;\n    \n    return aa * bb >> 32;\n}\nstatic uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)\n{\n    __uint128_t aa = a;\n    __uint128_t bb = b;\n    \n    return aa * bb >> 64;\n}\n#endif\n#if defined(__OPENCL_VERSION__)\nstatic uint8_t futrts_mad_hi8(uint8_t a, uint8_t b, uint8_t c)\n{\n    return mad_hi(a, b, c);\n}\nstatic uint16_t futrts_mad_hi16(uint16_t a, uint16_t b, uint16_t c)\n{\n    return mad_hi(a, b, c);\n}\nstatic uint32_t futrts_mad_hi32(uint32_t a, uint32_t b, uint32_t c)\n{\n    return mad_hi(a, b, c);\n}\nstatic uint64_t futrts_mad_hi64(uint64_t a, uint64_t b, uint64_t c)\n{\n    return mad_hi(a, b, c);\n}\n#else\nstatic uint8_t futrts_mad_hi8(uint8_t a, uint8_t b, uint8_t c)\n{\n    return futrts_mul_hi8(a, b) + c;\n}\nstatic uint16_t futrts_mad_hi16(uin",
                   "t16_t a, uint16_t b, uint16_t c)\n{\n    return futrts_mul_hi16(a, b) + c;\n}\nstatic uint32_t futrts_mad_hi32(uint32_t a, uint32_t b, uint32_t c)\n{\n    return futrts_mul_hi32(a, b) + c;\n}\nstatic uint64_t futrts_mad_hi64(uint64_t a, uint64_t b, uint64_t c)\n{\n    return futrts_mul_hi64(a, b) + c;\n}\n#endif\n#if defined(__OPENCL_VERSION__)\nstatic int32_t futrts_clzz8(int8_t x)\n{\n    return clz(x);\n}\nstatic int32_t futrts_clzz16(int16_t x)\n{\n    return clz(x);\n}\nstatic int32_t futrts_clzz32(int32_t x)\n{\n    return clz(x);\n}\nstatic int32_t futrts_clzz64(int64_t x)\n{\n    return clz(x);\n}\n#elif defined(__CUDA_ARCH__)\nstatic int32_t futrts_clzz8(int8_t x)\n{\n    return __clz(zext_i8_i32(x)) - 24;\n}\nstatic int32_t futrts_clzz16(int16_t x)\n{\n    return __clz(zext_i16_i32(x)) - 16;\n}\nstatic int32_t futrts_clzz32(int32_t x)\n{\n    return __clz(x);\n}\nstatic int32_t futrts_clzz64(int64_t x)\n{\n    return __clzll(x);\n}\n#else\nstatic int32_t futrts_clzz8(int8_t x)\n{\n    int n = 0;\n    int bits = sizeof(x) * 8;\n    \n    for (int i = 0; i < bits; i++) {\n        if (x < 0)\n            break;\n        n++;\n        x <<= 1;\n    }\n    return n;\n}\nstatic int32_t futrts_clzz16(int16_t x)\n{\n    int n = 0;\n    int bits = sizeof(x) * 8;\n    \n    for (int i = 0; i < bits; i++) {\n        if (x < 0)\n            break;\n        n++;\n        x <<= 1;\n    }\n    return n;\n}\nstatic int32_t futrts_clzz32(int32_t x)\n{\n    int n = 0;\n    int bits = sizeof(x) * 8;\n    \n    for (int i = 0; i < bits; i++) {\n        if (x < 0)\n            break;\n        n++;\n        x <<= 1;\n    }\n    return n;\n}\nstatic int32_t futrts_clzz64(int64_t x)\n{\n    int n = 0;\n    int bits = sizeof(x) * 8;\n    \n    for (int i = 0; i < bits; i++) {\n        if (x < 0)\n            break;\n        n++;\n        x <<= 1;\n    }\n    return n;\n}\n#endif\n#if defined(__OPENCL_VERSION__)\nstatic int32_t futrts_ctzz8(int8_t x)\n{\n    int i = 0;\n    \n    for (; i < 8 && (x & 1) == 0; i++, x >>= 1)\n        ;\n    return i;\n}\nstatic int32_t futrts_ctzz16(int16_t ",
                   "x)\n{\n    int i = 0;\n    \n    for (; i < 16 && (x & 1) == 0; i++, x >>= 1)\n        ;\n    return i;\n}\nstatic int32_t futrts_ctzz32(int32_t x)\n{\n    int i = 0;\n    \n    for (; i < 32 && (x & 1) == 0; i++, x >>= 1)\n        ;\n    return i;\n}\nstatic int32_t futrts_ctzz64(int64_t x)\n{\n    int i = 0;\n    \n    for (; i < 64 && (x & 1) == 0; i++, x >>= 1)\n        ;\n    return i;\n}\n#elif defined(__CUDA_ARCH__)\nstatic int32_t futrts_ctzz8(int8_t x)\n{\n    int y = __ffs(x);\n    \n    return y == 0 ? 8 : y - 1;\n}\nstatic int32_t futrts_ctzz16(int16_t x)\n{\n    int y = __ffs(x);\n    \n    return y == 0 ? 16 : y - 1;\n}\nstatic int32_t futrts_ctzz32(int32_t x)\n{\n    int y = __ffs(x);\n    \n    return y == 0 ? 32 : y - 1;\n}\nstatic int32_t futrts_ctzz64(int64_t x)\n{\n    int y = __ffsll(x);\n    \n    return y == 0 ? 64 : y - 1;\n}\n#else\nstatic int32_t futrts_ctzz8(int8_t x)\n{\n    return x == 0 ? 8 : __builtin_ctz((uint32_t) x);\n}\nstatic int32_t futrts_ctzz16(int16_t x)\n{\n    return x == 0 ? 16 : __builtin_ctz((uint32_t) x);\n}\nstatic int32_t futrts_ctzz32(int32_t x)\n{\n    return x == 0 ? 32 : __builtin_ctz(x);\n}\nstatic int32_t futrts_ctzz64(int64_t x)\n{\n    return x == 0 ? 64 : __builtin_ctzll(x);\n}\n#endif\nstatic inline float fdiv32(float x, float y)\n{\n    return x / y;\n}\nstatic inline float fadd32(float x, float y)\n{\n    return x + y;\n}\nstatic inline float fsub32(float x, float y)\n{\n    return x - y;\n}\nstatic inline float fmul32(float x, float y)\n{\n    return x * y;\n}\nstatic inline float fmin32(float x, float y)\n{\n    return fmin(x, y);\n}\nstatic inline float fmax32(float x, float y)\n{\n    return fmax(x, y);\n}\nstatic inline float fpow32(float x, float y)\n{\n    return pow(x, y);\n}\nstatic inline bool cmplt32(float x, float y)\n{\n    return x < y;\n}\nstatic inline bool cmple32(float x, float y)\n{\n    return x <= y;\n}\nstatic inline float sitofp_i8_f32(int8_t x)\n{\n    return (float) x;\n}\nstatic inline float sitofp_i16_f32(int16_t x)\n{\n    return (float) x;\n}\nstatic inline float sitofp_i32_f32(int32_t x",
                   ")\n{\n    return (float) x;\n}\nstatic inline float sitofp_i64_f32(int64_t x)\n{\n    return (float) x;\n}\nstatic inline float uitofp_i8_f32(uint8_t x)\n{\n    return (float) x;\n}\nstatic inline float uitofp_i16_f32(uint16_t x)\n{\n    return (float) x;\n}\nstatic inline float uitofp_i32_f32(uint32_t x)\n{\n    return (float) x;\n}\nstatic inline float uitofp_i64_f32(uint64_t x)\n{\n    return (float) x;\n}\nstatic inline int8_t fptosi_f32_i8(float x)\n{\n    return (int8_t) x;\n}\nstatic inline int16_t fptosi_f32_i16(float x)\n{\n    return (int16_t) x;\n}\nstatic inline int32_t fptosi_f32_i32(float x)\n{\n    return (int32_t) x;\n}\nstatic inline int64_t fptosi_f32_i64(float x)\n{\n    return (int64_t) x;\n}\nstatic inline uint8_t fptoui_f32_i8(float x)\n{\n    return (uint8_t) x;\n}\nstatic inline uint16_t fptoui_f32_i16(float x)\n{\n    return (uint16_t) x;\n}\nstatic inline uint32_t fptoui_f32_i32(float x)\n{\n    return (uint32_t) x;\n}\nstatic inline uint64_t fptoui_f32_i64(float x)\n{\n    return (uint64_t) x;\n}\nstatic inline float futrts_log32(float x)\n{\n    return log(x);\n}\nstatic inline float futrts_log2_32(float x)\n{\n    return log2(x);\n}\nstatic inline float futrts_log10_32(float x)\n{\n    return log10(x);\n}\nstatic inline float futrts_sqrt32(float x)\n{\n    return sqrt(x);\n}\nstatic inline float futrts_exp32(float x)\n{\n    return exp(x);\n}\nstatic inline float futrts_cos32(float x)\n{\n    return cos(x);\n}\nstatic inline float futrts_sin32(float x)\n{\n    return sin(x);\n}\nstatic inline float futrts_tan32(float x)\n{\n    return tan(x);\n}\nstatic inline float futrts_acos32(float x)\n{\n    return acos(x);\n}\nstatic inline float futrts_asin32(float x)\n{\n    return asin(x);\n}\nstatic inline float futrts_atan32(float x)\n{\n    return atan(x);\n}\nstatic inline float futrts_cosh32(float x)\n{\n    return cosh(x);\n}\nstatic inline float futrts_sinh32(float x)\n{\n    return sinh(x);\n}\nstatic inline float futrts_tanh32(float x)\n{\n    return tanh(x);\n}\nstatic inline float futrts_acosh32(float x)\n{\n    return acosh(x);\n}\nstatic inline f",
                   "loat futrts_asinh32(float x)\n{\n    return asinh(x);\n}\nstatic inline float futrts_atanh32(float x)\n{\n    return atanh(x);\n}\nstatic inline float futrts_atan2_32(float x, float y)\n{\n    return atan2(x, y);\n}\nstatic inline float futrts_gamma32(float x)\n{\n    return tgamma(x);\n}\nstatic inline float futrts_lgamma32(float x)\n{\n    return lgamma(x);\n}\nstatic inline bool futrts_isnan32(float x)\n{\n    return isnan(x);\n}\nstatic inline bool futrts_isinf32(float x)\n{\n    return isinf(x);\n}\nstatic inline int32_t futrts_to_bits32(float x)\n{\n    union {\n        float f;\n        int32_t t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\nstatic inline float futrts_from_bits32(int32_t x)\n{\n    union {\n        int32_t f;\n        float t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\n#ifdef __OPENCL_VERSION__\nstatic inline float fmod32(float x, float y)\n{\n    return fmod(x, y);\n}\nstatic inline float futrts_round32(float x)\n{\n    return rint(x);\n}\nstatic inline float futrts_floor32(float x)\n{\n    return floor(x);\n}\nstatic inline float futrts_ceil32(float x)\n{\n    return ceil(x);\n}\nstatic inline float futrts_lerp32(float v0, float v1, float t)\n{\n    return mix(v0, v1, t);\n}\nstatic inline float futrts_mad32(float a, float b, float c)\n{\n    return mad(a, b, c);\n}\nstatic inline float futrts_fma32(float a, float b, float c)\n{\n    return fma(a, b, c);\n}\n#else\nstatic inline float fmod32(float x, float y)\n{\n    return fmodf(x, y);\n}\nstatic inline float futrts_round32(float x)\n{\n    return rintf(x);\n}\nstatic inline float futrts_floor32(float x)\n{\n    return floorf(x);\n}\nstatic inline float futrts_ceil32(float x)\n{\n    return ceilf(x);\n}\nstatic inline float futrts_lerp32(float v0, float v1, float t)\n{\n    return v0 + (v1 - v0) * t;\n}\nstatic inline float futrts_mad32(float a, float b, float c)\n{\n    return a * b + c;\n}\nstatic inline float futrts_fma32(float a, float b, float c)\n{\n    return fmaf(a, b, c);\n}\n#endif\n// Start of atomics.h\n\ninline int32_t atomic_add_i32_global(volatile __global int32_t *p,",
                   " int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicAdd((int32_t*)p, x);\n#else\n  return atomic_add(p, x);\n#endif\n}\n\ninline int32_t atomic_add_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicAdd((int32_t*)p, x);\n#else\n  return atomic_add(p, x);\n#endif\n}\n\ninline float atomic_fadd_f32_global(volatile __global float *p, float x) {\n#ifdef FUTHARK_CUDA\n  return atomicAdd((float*)p, x);\n#else\n  union { int32_t i; float f; } old;\n  union { int32_t i; float f; } assumed;\n  old.f = *p;\n  do {\n    assumed.f = old.f;\n    old.f = old.f + x;\n    old.i = atomic_cmpxchg((volatile __global int32_t*)p, assumed.i, old.i);\n  } while (assumed.i != old.i);\n  return old.f;\n#endif\n}\n\ninline float atomic_fadd_f32_local(volatile __local float *p, float x) {\n#ifdef FUTHARK_CUDA\n  return atomicAdd((float*)p, x);\n#else\n  union { int32_t i; float f; } old;\n  union { int32_t i; float f; } assumed;\n  old.f = *p;\n  do {\n    assumed.f = old.f;\n    old.f = old.f + x;\n    old.i = atomic_cmpxchg((volatile __local int32_t*)p, assumed.i, old.i);\n  } while (assumed.i != old.i);\n  return old.f;\n#endif\n}\n\ninline int32_t atomic_smax_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMax((int32_t*)p, x);\n#else\n  return atomic_max(p, x);\n#endif\n}\n\ninline int32_t atomic_smax_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMax((int32_t*)p, x);\n#else\n  return atomic_max(p, x);\n#endif\n}\n\ninline int32_t atomic_smin_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMin((int32_t*)p, x);\n#else\n  return atomic_min(p, x);\n#endif\n}\n\ninline int32_t atomic_smin_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMin((int32_t*)p, x);\n#else\n  return atomic_min(p, x);\n#endif\n}\n\ninline uint32_t atomic_umax_i32_global(volatile __global uint32_t *p, uint32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMax((uint32_t*)p, x);\n#else\n  return atomic_max(",
                   "p, x);\n#endif\n}\n\ninline uint32_t atomic_umax_i32_local(volatile __local uint32_t *p, uint32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMax((uint32_t*)p, x);\n#else\n  return atomic_max(p, x);\n#endif\n}\n\ninline uint32_t atomic_umin_i32_global(volatile __global uint32_t *p, uint32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMin((uint32_t*)p, x);\n#else\n  return atomic_min(p, x);\n#endif\n}\n\ninline uint32_t atomic_umin_i32_local(volatile __local uint32_t *p, uint32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMin((uint32_t*)p, x);\n#else\n  return atomic_min(p, x);\n#endif\n}\n\ninline int32_t atomic_and_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicAnd((int32_t*)p, x);\n#else\n  return atomic_and(p, x);\n#endif\n}\n\ninline int32_t atomic_and_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicAnd((int32_t*)p, x);\n#else\n  return atomic_and(p, x);\n#endif\n}\n\ninline int32_t atomic_or_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicOr((int32_t*)p, x);\n#else\n  return atomic_or(p, x);\n#endif\n}\n\ninline int32_t atomic_or_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicOr((int32_t*)p, x);\n#else\n  return atomic_or(p, x);\n#endif\n}\n\ninline int32_t atomic_xor_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicXor((int32_t*)p, x);\n#else\n  return atomic_xor(p, x);\n#endif\n}\n\ninline int32_t atomic_xor_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicXor((int32_t*)p, x);\n#else\n  return atomic_xor(p, x);\n#endif\n}\n\ninline int32_t atomic_xchg_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicExch((int32_t*)p, x);\n#else\n  return atomic_xor(p, x);\n#endif\n}\n\ninline int32_t atomic_xchg_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicExch((int32_t*)p, x);\n#else\n  return atomic_xor(p, x);\n#endif\n}\n\ninline int32_",
                   "t atomic_cmpxchg_i32_global(volatile __global int32_t *p,\n                                         int32_t cmp, int32_t val) {\n#ifdef FUTHARK_CUDA\n  return atomicCAS((int32_t*)p, cmp, val);\n#else\n  return atomic_cmpxchg(p, cmp, val);\n#endif\n}\n\ninline int32_t atomic_cmpxchg_i32_local(volatile __local int32_t *p,\n                                         int32_t cmp, int32_t val) {\n#ifdef FUTHARK_CUDA\n  return atomicCAS((int32_t*)p, cmp, val);\n#else\n  return atomic_cmpxchg(p, cmp, val);\n#endif\n}\n\n// End of atomics.h\n\n\n\n\n__kernel void mainziscan_stage1_6376(__global int *global_failure,\n                                     __local volatile\n                                     int64_t *scan_arr_mem_6444_backing_aligned_0,\n                                     __local volatile\n                                     int64_t *scan_arr_mem_6442_backing_aligned_1,\n                                     __local volatile\n                                     int64_t *scan_arr_mem_6440_backing_aligned_2,\n                                     __local volatile\n                                     int64_t *scan_arr_mem_6438_backing_aligned_3,\n                                     int64_t implz2080U_6265,\n                                     int32_t defunc_2_radix_sort_step_arg_6280,\n                                     int32_t defunc_0_get_bit_arg_6281,\n                                     unsigned char cond_6282,\n                                     unsigned char cond_6283,\n                                     int32_t num_threads_6432, __global\n                                     unsigned char *mem_param_6391, __global\n                                     unsigned char *mem_6396, __global\n                                     unsigned char *mem_6398, __global\n                                     unsigned char *mem_6400, __global\n                                     unsigned char *mem_6402, __global\n                                     unsigned char *mem_6404)\n{\n    #define segscan_group_s",
                   "izze_6371 (mainzisegscan_group_sizze_6370)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict scan_arr_mem_6444_backing_3 =\n                          (__local volatile\n                           char *) scan_arr_mem_6444_backing_aligned_0;\n    __local volatile char *restrict scan_arr_mem_6442_backing_2 =\n                          (__local volatile\n                           char *) scan_arr_mem_6442_backing_aligned_1;\n    __local volatile char *restrict scan_arr_mem_6440_backing_1 =\n                          (__local volatile\n                           char *) scan_arr_mem_6440_backing_aligned_2;\n    __local volatile char *restrict scan_arr_mem_6438_backing_0 =\n                          (__local volatile\n                           char *) scan_arr_mem_6438_backing_aligned_3;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_6433;\n    int32_t local_tid_6434;\n    int64_t group_sizze_6437;\n    int32_t wave_sizze_6436;\n    int32_t group_tid_6435;\n    \n    global_tid_6433 = get_global_id(0);\n    local_tid_6434 = get_local_id(0);\n    group_sizze_6437 = get_local_size(0);\n    wave_sizze_6436 = LOCKSTEP_WIDTH;\n    group_tid_6435 = get_group_id(0);\n    \n    int32_t phys_tid_6376;\n    \n    phys_tid_6376 = global_tid_6433;\n    \n    __local char *scan_arr_mem_6438;\n    __local char *scan_arr_mem_6440;\n    __local char *scan_arr_mem_6442;\n    __local char *scan_arr_mem_6444;\n    \n    scan_arr_mem_6438 = (__local char *) scan_arr_mem_6438_backing_0;\n    scan_arr_mem_6440 = (__local char *) scan_arr_mem_6440_backing_1;\n    scan_arr_mem_6442 = (__local char *) scan_arr_mem_6442_backing_2;\n    scan_arr_mem_6444 = (__local char *) scan_arr_mem_6444_backing_3;\n    \n    int64_t x_6301;\n    int64_t x_6302;\n    int64_t x_6303;\n    int64_t x_6304;\n    int64_t x_6305;\n    int64_t x_6306;\n    int64_t x_6307;\n    int64_t x_6308;\n    \n    x_6301 = (int64_t) 0;\n    x_6302 = (int64_t) 0;",
                   "\n    x_6303 = (int64_t) 0;\n    x_6304 = (int64_t) 0;\n    for (int64_t j_6446 = 0; j_6446 < sdiv_up64(implz2080U_6265,\n                                                sext_i32_i64(num_threads_6432));\n         j_6446++) {\n        int64_t chunk_offset_6447 = segscan_group_sizze_6371 * j_6446 +\n                sext_i32_i64(group_tid_6435) * (segscan_group_sizze_6371 *\n                                                sdiv_up64(implz2080U_6265,\n                                                          sext_i32_i64(num_threads_6432)));\n        int64_t flat_idx_6448 = chunk_offset_6447 +\n                sext_i32_i64(local_tid_6434);\n        int64_t gtid_6375 = flat_idx_6448;\n        \n        // threads in bounds read input\n        {\n            if (slt64(gtid_6375, implz2080U_6265)) {\n                int32_t x_6313 = ((__global\n                                   int32_t *) mem_param_6391)[gtid_6375];\n                int32_t zgzg_res_6314 = ashr32(x_6313,\n                                               defunc_0_get_bit_arg_6281);\n                int32_t za_res_6315 = 1 & zgzg_res_6314;\n                int32_t defunc_1_get_bit_res_6316;\n                \n                if (cond_6282) {\n                    int32_t defunc_1_get_bit_res_t_res_6317 = 1 ^ za_res_6315;\n                    \n                    defunc_1_get_bit_res_6316 = defunc_1_get_bit_res_t_res_6317;\n                } else {\n                    defunc_1_get_bit_res_6316 = za_res_6315;\n                }\n                \n                int32_t x_6318 = mul32(2, defunc_1_get_bit_res_6316);\n                int32_t zgzg_res_6319 = ashr32(x_6313,\n                                               defunc_2_radix_sort_step_arg_6280);\n                int32_t za_res_6320 = 1 & zgzg_res_6319;\n                int32_t defunc_1_get_bit_res_6321;\n                \n                if (cond_6283) {\n                    int32_t defunc_1_get_bit_res_t_res_6322 = 1 ^ za_res_6320;\n                    \n                    defunc_1_get_bit_re",
                   "s_6321 = defunc_1_get_bit_res_t_res_6322;\n                } else {\n                    defunc_1_get_bit_res_6321 = za_res_6320;\n                }\n                \n                int32_t defunc_0_f_res_6323 = add32(x_6318,\n                                                    defunc_1_get_bit_res_6321);\n                bool cond_6324 = defunc_0_f_res_6323 == 0;\n                int64_t defunc_0_f_res_6325 = btoi_bool_i64(cond_6324);\n                int64_t defunc_0_f_res_6326;\n                int64_t defunc_0_f_res_6327;\n                int64_t defunc_0_f_res_6328;\n                \n                if (cond_6324) {\n                    defunc_0_f_res_6326 = (int64_t) 0;\n                    defunc_0_f_res_6327 = (int64_t) 0;\n                    defunc_0_f_res_6328 = (int64_t) 0;\n                } else {\n                    bool cond_6329 = defunc_0_f_res_6323 == 1;\n                    int64_t defunc_0_f_res_f_res_6330 =\n                            btoi_bool_i64(cond_6329);\n                    int64_t defunc_0_f_res_f_res_6331;\n                    int64_t defunc_0_f_res_f_res_6332;\n                    \n                    if (cond_6329) {\n                        defunc_0_f_res_f_res_6331 = (int64_t) 0;\n                        defunc_0_f_res_f_res_6332 = (int64_t) 0;\n                    } else {\n                        bool cond_6333 = defunc_0_f_res_6323 == 2;\n                        int64_t defunc_0_f_res_f_res_f_res_6334 =\n                                btoi_bool_i64(cond_6333);\n                        bool cond_neg_6335 = !cond_6333;\n                        int64_t defunc_0_f_res_f_res_f_res_6336 =\n                                btoi_bool_i64(cond_neg_6335);\n                        \n                        defunc_0_f_res_f_res_6331 =\n                            defunc_0_f_res_f_res_f_res_6334;\n                        defunc_0_f_res_f_res_6332 =\n                            defunc_0_f_res_f_res_f_res_6336;\n                    }\n                    defunc_0_f_res_6326 = ",
                   "defunc_0_f_res_f_res_6330;\n                    defunc_0_f_res_6327 = defunc_0_f_res_f_res_6331;\n                    defunc_0_f_res_6328 = defunc_0_f_res_f_res_6332;\n                }\n                // write to-scan values to parameters\n                {\n                    x_6305 = defunc_0_f_res_6325;\n                    x_6306 = defunc_0_f_res_6326;\n                    x_6307 = defunc_0_f_res_6327;\n                    x_6308 = defunc_0_f_res_6328;\n                }\n                // write mapped values results to global memory\n                {\n                    ((__global int32_t *) mem_6404)[gtid_6375] =\n                        defunc_0_f_res_6323;\n                }\n            }\n        }\n        // do one intra-group scan operation\n        {\n            // maybe restore some to-scan values to parameters, or read neutral\n            {\n                if (!slt64(gtid_6375, implz2080U_6265)) {\n                    x_6305 = (int64_t) 0;\n                    x_6306 = (int64_t) 0;\n                    x_6307 = (int64_t) 0;\n                    x_6308 = (int64_t) 0;\n                }\n            }\n            // combine with carry and write to local memory\n            {\n                int64_t defunc_1_op_res_6309 = add64(x_6301, x_6305);\n                int64_t defunc_1_op_res_6310 = add64(x_6302, x_6306);\n                int64_t defunc_1_op_res_6311 = add64(x_6303, x_6307);\n                int64_t defunc_1_op_res_6312 = add64(x_6304, x_6308);\n                \n                ((__local\n                  int64_t *) scan_arr_mem_6438)[sext_i32_i64(local_tid_6434)] =\n                    defunc_1_op_res_6309;\n                ((__local\n                  int64_t *) scan_arr_mem_6440)[sext_i32_i64(local_tid_6434)] =\n                    defunc_1_op_res_6310;\n                ((__local\n                  int64_t *) scan_arr_mem_6442)[sext_i32_i64(local_tid_6434)] =\n                    defunc_1_op_res_6311;\n                ((__local\n                  int64_t *) scan_arr_mem_644",
                   "4)[sext_i32_i64(local_tid_6434)] =\n                    defunc_1_op_res_6312;\n            }\n            barrier(CLK_LOCAL_MEM_FENCE);\n            \n            int64_t x_6449;\n            int64_t x_6450;\n            int64_t x_6451;\n            int64_t x_6452;\n            int64_t x_6453;\n            int64_t x_6454;\n            int64_t x_6455;\n            int64_t x_6456;\n            int64_t x_6461;\n            int64_t x_6462;\n            int64_t x_6463;\n            int64_t x_6464;\n            int64_t x_6465;\n            int64_t x_6466;\n            int64_t x_6467;\n            int64_t x_6468;\n            bool ltid_in_bounds_6473;\n            \n            ltid_in_bounds_6473 = slt64(sext_i32_i64(local_tid_6434),\n                                        segscan_group_sizze_6371);\n            \n            int32_t skip_threads_6474;\n            \n            // read input for in-block scan\n            {\n                if (ltid_in_bounds_6473) {\n                    x_6453 = ((volatile __local\n                               int64_t *) scan_arr_mem_6438)[sext_i32_i64(local_tid_6434)];\n                    x_6454 = ((volatile __local\n                               int64_t *) scan_arr_mem_6440)[sext_i32_i64(local_tid_6434)];\n                    x_6455 = ((volatile __local\n                               int64_t *) scan_arr_mem_6442)[sext_i32_i64(local_tid_6434)];\n                    x_6456 = ((volatile __local\n                               int64_t *) scan_arr_mem_6444)[sext_i32_i64(local_tid_6434)];\n                    if ((local_tid_6434 - squot32(local_tid_6434, 32) * 32) ==\n                        0) {\n                        x_6449 = x_6453;\n                        x_6450 = x_6454;\n                        x_6451 = x_6455;\n                        x_6452 = x_6456;\n                    }\n                }\n            }\n            // in-block scan (hopefully no barriers needed)\n            {\n                skip_threads_6474 = 1;\n                while (slt32(skip_threads_6474, 32)) ",
                   "{\n                    if (sle32(skip_threads_6474, local_tid_6434 -\n                              squot32(local_tid_6434, 32) * 32) &&\n                        ltid_in_bounds_6473) {\n                        // read operands\n                        {\n                            x_6449 = ((volatile __local\n                                       int64_t *) scan_arr_mem_6438)[sext_i32_i64(local_tid_6434) -\n                                                                     sext_i32_i64(skip_threads_6474)];\n                            x_6450 = ((volatile __local\n                                       int64_t *) scan_arr_mem_6440)[sext_i32_i64(local_tid_6434) -\n                                                                     sext_i32_i64(skip_threads_6474)];\n                            x_6451 = ((volatile __local\n                                       int64_t *) scan_arr_mem_6442)[sext_i32_i64(local_tid_6434) -\n                                                                     sext_i32_i64(skip_threads_6474)];\n                            x_6452 = ((volatile __local\n                                       int64_t *) scan_arr_mem_6444)[sext_i32_i64(local_tid_6434) -\n                                                                     sext_i32_i64(skip_threads_6474)];\n                        }\n                        // perform operation\n                        {\n                            int64_t defunc_1_op_res_6457 = add64(x_6449,\n                                                                 x_6453);\n                            int64_t defunc_1_op_res_6458 = add64(x_6450,\n                                                                 x_6454);\n                            int64_t defunc_1_op_res_6459 = add64(x_6451,\n                                                                 x_6455);\n                            int64_t defunc_1_op_res_6460 = add64(x_6452,\n                                                                 x_6456);\n                            \n ",
                   "                           x_6449 = defunc_1_op_res_6457;\n                            x_6450 = defunc_1_op_res_6458;\n                            x_6451 = defunc_1_op_res_6459;\n                            x_6452 = defunc_1_op_res_6460;\n                        }\n                    }\n                    if (sle32(wave_sizze_6436, skip_threads_6474)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    if (sle32(skip_threads_6474, local_tid_6434 -\n                              squot32(local_tid_6434, 32) * 32) &&\n                        ltid_in_bounds_6473) {\n                        // write result\n                        {\n                            ((volatile __local\n                              int64_t *) scan_arr_mem_6438)[sext_i32_i64(local_tid_6434)] =\n                                x_6449;\n                            x_6453 = x_6449;\n                            ((volatile __local\n                              int64_t *) scan_arr_mem_6440)[sext_i32_i64(local_tid_6434)] =\n                                x_6450;\n                            x_6454 = x_6450;\n                            ((volatile __local\n                              int64_t *) scan_arr_mem_6442)[sext_i32_i64(local_tid_6434)] =\n                                x_6451;\n                            x_6455 = x_6451;\n                            ((volatile __local\n                              int64_t *) scan_arr_mem_6444)[sext_i32_i64(local_tid_6434)] =\n                                x_6452;\n                            x_6456 = x_6452;\n                        }\n                    }\n                    if (sle32(wave_sizze_6436, skip_threads_6474)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    skip_threads_6474 *= 2;\n                }\n            }\n            barrier(CLK_LOCAL_MEM_FENCE);\n            // last thread of block 'i' writes its result to offset 'i'\n            {\n                if ((local_tid_64",
                   "34 - squot32(local_tid_6434, 32) * 32) == 31 &&\n                    ltid_in_bounds_6473) {\n                    ((volatile __local\n                      int64_t *) scan_arr_mem_6438)[sext_i32_i64(squot32(local_tid_6434,\n                                                                         32))] =\n                        x_6449;\n                    ((volatile __local\n                      int64_t *) scan_arr_mem_6440)[sext_i32_i64(squot32(local_tid_6434,\n                                                                         32))] =\n                        x_6450;\n                    ((volatile __local\n                      int64_t *) scan_arr_mem_6442)[sext_i32_i64(squot32(local_tid_6434,\n                                                                         32))] =\n                        x_6451;\n                    ((volatile __local\n                      int64_t *) scan_arr_mem_6444)[sext_i32_i64(squot32(local_tid_6434,\n                                                                         32))] =\n                        x_6452;\n                }\n            }\n            barrier(CLK_LOCAL_MEM_FENCE);\n            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'\n            {\n                int32_t skip_threads_6475;\n                \n                // read input for in-block scan\n                {\n                    if (squot32(local_tid_6434, 32) == 0 &&\n                        ltid_in_bounds_6473) {\n                        x_6465 = ((volatile __local\n                                   int64_t *) scan_arr_mem_6438)[sext_i32_i64(local_tid_6434)];\n                        x_6466 = ((volatile __local\n                                   int64_t *) scan_arr_mem_6440)[sext_i32_i64(local_tid_6434)];\n                        x_6467 = ((volatile __local\n                                   int64_t *) scan_arr_mem_6442)[sext_i32_i64(local_tid_6434)];\n                        x_6468 = ((volatile __local\n                                  ",
                   " int64_t *) scan_arr_mem_6444)[sext_i32_i64(local_tid_6434)];\n                        if ((local_tid_6434 - squot32(local_tid_6434, 32) *\n                             32) == 0) {\n                            x_6461 = x_6465;\n                            x_6462 = x_6466;\n                            x_6463 = x_6467;\n                            x_6464 = x_6468;\n                        }\n                    }\n                }\n                // in-block scan (hopefully no barriers needed)\n                {\n                    skip_threads_6475 = 1;\n                    while (slt32(skip_threads_6475, 32)) {\n                        if (sle32(skip_threads_6475, local_tid_6434 -\n                                  squot32(local_tid_6434, 32) * 32) &&\n                            (squot32(local_tid_6434, 32) == 0 &&\n                             ltid_in_bounds_6473)) {\n                            // read operands\n                            {\n                                x_6461 = ((volatile __local\n                                           int64_t *) scan_arr_mem_6438)[sext_i32_i64(local_tid_6434) -\n                                                                         sext_i32_i64(skip_threads_6475)];\n                                x_6462 = ((volatile __local\n                                           int64_t *) scan_arr_mem_6440)[sext_i32_i64(local_tid_6434) -\n                                                                         sext_i32_i64(skip_threads_6475)];\n                                x_6463 = ((volatile __local\n                                           int64_t *) scan_arr_mem_6442)[sext_i32_i64(local_tid_6434) -\n                                                                         sext_i32_i64(skip_threads_6475)];\n                                x_6464 = ((volatile __local\n                                           int64_t *) scan_arr_mem_6444)[sext_i32_i64(local_tid_6434) -\n                                                                         sext_i3",
                   "2_i64(skip_threads_6475)];\n                            }\n                            // perform operation\n                            {\n                                int64_t defunc_1_op_res_6469 = add64(x_6461,\n                                                                     x_6465);\n                                int64_t defunc_1_op_res_6470 = add64(x_6462,\n                                                                     x_6466);\n                                int64_t defunc_1_op_res_6471 = add64(x_6463,\n                                                                     x_6467);\n                                int64_t defunc_1_op_res_6472 = add64(x_6464,\n                                                                     x_6468);\n                                \n                                x_6461 = defunc_1_op_res_6469;\n                                x_6462 = defunc_1_op_res_6470;\n                                x_6463 = defunc_1_op_res_6471;\n                                x_6464 = defunc_1_op_res_6472;\n                            }\n                        }\n                        if (sle32(wave_sizze_6436, skip_threads_6475)) {\n                            barrier(CLK_LOCAL_MEM_FENCE);\n                        }\n                        if (sle32(skip_threads_6475, local_tid_6434 -\n                                  squot32(local_tid_6434, 32) * 32) &&\n                            (squot32(local_tid_6434, 32) == 0 &&\n                             ltid_in_bounds_6473)) {\n                            // write result\n                            {\n                                ((volatile __local\n                                  int64_t *) scan_arr_mem_6438)[sext_i32_i64(local_tid_6434)] =\n                                    x_6461;\n                                x_6465 = x_6461;\n                                ((volatile __local\n                                  int64_t *) scan_arr_mem_6440)[sext_i32_i64(local_tid_6434)] =\n                          ",
                   "          x_6462;\n                                x_6466 = x_6462;\n                                ((volatile __local\n                                  int64_t *) scan_arr_mem_6442)[sext_i32_i64(local_tid_6434)] =\n                                    x_6463;\n                                x_6467 = x_6463;\n                                ((volatile __local\n                                  int64_t *) scan_arr_mem_6444)[sext_i32_i64(local_tid_6434)] =\n                                    x_6464;\n                                x_6468 = x_6464;\n                            }\n                        }\n                        if (sle32(wave_sizze_6436, skip_threads_6475)) {\n                            barrier(CLK_LOCAL_MEM_FENCE);\n                        }\n                        skip_threads_6475 *= 2;\n                    }\n                }\n            }\n            barrier(CLK_LOCAL_MEM_FENCE);\n            // carry-in for every block except the first\n            {\n                if (!(squot32(local_tid_6434, 32) == 0 ||\n                      !ltid_in_bounds_6473)) {\n                    // read operands\n                    {\n                        x_6453 = x_6449;\n                        x_6454 = x_6450;\n                        x_6455 = x_6451;\n                        x_6456 = x_6452;\n                        x_6449 = ((__local\n                                   int64_t *) scan_arr_mem_6438)[sext_i32_i64(squot32(local_tid_6434,\n                                                                                      32)) -\n                                                                 (int64_t) 1];\n                        x_6450 = ((__local\n                                   int64_t *) scan_arr_mem_6440)[sext_i32_i64(squot32(local_tid_6434,\n                                                                                      32)) -\n                                                                 (int64_t) 1];\n                        x_6451 = ((__local\n                    ",
                   "               int64_t *) scan_arr_mem_6442)[sext_i32_i64(squot32(local_tid_6434,\n                                                                                      32)) -\n                                                                 (int64_t) 1];\n                        x_6452 = ((__local\n                                   int64_t *) scan_arr_mem_6444)[sext_i32_i64(squot32(local_tid_6434,\n                                                                                      32)) -\n                                                                 (int64_t) 1];\n                    }\n                    // perform operation\n                    {\n                        int64_t defunc_1_op_res_6457 = add64(x_6449, x_6453);\n                        int64_t defunc_1_op_res_6458 = add64(x_6450, x_6454);\n                        int64_t defunc_1_op_res_6459 = add64(x_6451, x_6455);\n                        int64_t defunc_1_op_res_6460 = add64(x_6452, x_6456);\n                        \n                        x_6449 = defunc_1_op_res_6457;\n                        x_6450 = defunc_1_op_res_6458;\n                        x_6451 = defunc_1_op_res_6459;\n                        x_6452 = defunc_1_op_res_6460;\n                    }\n                    // write final result\n                    {\n                        ((__local\n                          int64_t *) scan_arr_mem_6438)[sext_i32_i64(local_tid_6434)] =\n                            x_6449;\n                        ((__local\n                          int64_t *) scan_arr_mem_6440)[sext_i32_i64(local_tid_6434)] =\n                            x_6450;\n                        ((__local\n                          int64_t *) scan_arr_mem_6442)[sext_i32_i64(local_tid_6434)] =\n                            x_6451;\n                        ((__local\n                          int64_t *) scan_arr_mem_6444)[sext_i32_i64(local_tid_6434)] =\n                            x_6452;\n                    }\n                }\n            }\n            bar",
                   "rier(CLK_LOCAL_MEM_FENCE);\n            // restore correct values for first block\n            {\n                if (squot32(local_tid_6434, 32) == 0) {\n                    ((__local\n                      int64_t *) scan_arr_mem_6438)[sext_i32_i64(local_tid_6434)] =\n                        x_6453;\n                    ((__local\n                      int64_t *) scan_arr_mem_6440)[sext_i32_i64(local_tid_6434)] =\n                        x_6454;\n                    ((__local\n                      int64_t *) scan_arr_mem_6442)[sext_i32_i64(local_tid_6434)] =\n                        x_6455;\n                    ((__local\n                      int64_t *) scan_arr_mem_6444)[sext_i32_i64(local_tid_6434)] =\n                        x_6456;\n                }\n            }\n            barrier(CLK_LOCAL_MEM_FENCE);\n            // threads in bounds write partial scan result\n            {\n                if (slt64(gtid_6375, implz2080U_6265)) {\n                    ((__global int64_t *) mem_6396)[gtid_6375] = ((__local\n                                                                   int64_t *) scan_arr_mem_6438)[sext_i32_i64(local_tid_6434)];\n                    ((__global int64_t *) mem_6398)[gtid_6375] = ((__local\n                                                                   int64_t *) scan_arr_mem_6440)[sext_i32_i64(local_tid_6434)];\n                    ((__global int64_t *) mem_6400)[gtid_6375] = ((__local\n                                                                   int64_t *) scan_arr_mem_6442)[sext_i32_i64(local_tid_6434)];\n                    ((__global int64_t *) mem_6402)[gtid_6375] = ((__local\n                                                                   int64_t *) scan_arr_mem_6444)[sext_i32_i64(local_tid_6434)];\n                }\n            }\n            barrier(CLK_LOCAL_MEM_FENCE);\n            // first thread reads last element as carry-in for next iteration\n            {\n                bool crosses_segment_6476 = 0;\n                bool should_load_car",
                   "ry_6477 = local_tid_6434 == 0 &&\n                     !crosses_segment_6476;\n                \n                if (should_load_carry_6477) {\n                    x_6301 = ((__local\n                               int64_t *) scan_arr_mem_6438)[segscan_group_sizze_6371 -\n                                                             (int64_t) 1];\n                    x_6302 = ((__local\n                               int64_t *) scan_arr_mem_6440)[segscan_group_sizze_6371 -\n                                                             (int64_t) 1];\n                    x_6303 = ((__local\n                               int64_t *) scan_arr_mem_6442)[segscan_group_sizze_6371 -\n                                                             (int64_t) 1];\n                    x_6304 = ((__local\n                               int64_t *) scan_arr_mem_6444)[segscan_group_sizze_6371 -\n                                                             (int64_t) 1];\n                }\n                if (!should_load_carry_6477) {\n                    x_6301 = (int64_t) 0;\n                    x_6302 = (int64_t) 0;\n                    x_6303 = (int64_t) 0;\n                    x_6304 = (int64_t) 0;\n                }\n            }\n            barrier(CLK_LOCAL_MEM_FENCE);\n        }\n    }\n    \n  error_1:\n    return;\n    #undef segscan_group_sizze_6371\n}\n__kernel void mainziscan_stage2_6376(__global int *global_failure,\n                                     __local volatile\n                                     int64_t *scan_arr_mem_6489_backing_aligned_0,\n                                     __local volatile\n                                     int64_t *scan_arr_mem_6487_backing_aligned_1,\n                                     __local volatile\n                                     int64_t *scan_arr_mem_6485_backing_aligned_2,\n                                     __local volatile\n                                     int64_t *scan_arr_mem_6483_backing_aligned_3,\n                                     int64_t imp",
                   "lz2080U_6265,\n                                     int64_t stage1_num_groups_6431,\n                                     int32_t num_threads_6432, __global\n                                     unsigned char *mem_6396, __global\n                                     unsigned char *mem_6398, __global\n                                     unsigned char *mem_6400, __global\n                                     unsigned char *mem_6402)\n{\n    #define segscan_group_sizze_6371 (mainzisegscan_group_sizze_6370)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict scan_arr_mem_6489_backing_3 =\n                          (__local volatile\n                           char *) scan_arr_mem_6489_backing_aligned_0;\n    __local volatile char *restrict scan_arr_mem_6487_backing_2 =\n                          (__local volatile\n                           char *) scan_arr_mem_6487_backing_aligned_1;\n    __local volatile char *restrict scan_arr_mem_6485_backing_1 =\n                          (__local volatile\n                           char *) scan_arr_mem_6485_backing_aligned_2;\n    __local volatile char *restrict scan_arr_mem_6483_backing_0 =\n                          (__local volatile\n                           char *) scan_arr_mem_6483_backing_aligned_3;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_6478;\n    int32_t local_tid_6479;\n    int64_t group_sizze_6482;\n    int32_t wave_sizze_6481;\n    int32_t group_tid_6480;\n    \n    global_tid_6478 = get_global_id(0);\n    local_tid_6479 = get_local_id(0);\n    group_sizze_6482 = get_local_size(0);\n    wave_sizze_6481 = LOCKSTEP_WIDTH;\n    group_tid_6480 = get_group_id(0);\n    \n    int32_t phys_tid_6376;\n    \n    phys_tid_6376 = global_tid_6478;\n    \n    __local char *scan_arr_mem_6483;\n    __local char *scan_arr_mem_6485;\n    __local char *scan_arr_mem_6487;\n    __local char *scan_arr_mem_6489;\n    \n    scan_arr_mem_6483 = (__local char *) s",
                   "can_arr_mem_6483_backing_0;\n    scan_arr_mem_6485 = (__local char *) scan_arr_mem_6485_backing_1;\n    scan_arr_mem_6487 = (__local char *) scan_arr_mem_6487_backing_2;\n    scan_arr_mem_6489 = (__local char *) scan_arr_mem_6489_backing_3;\n    \n    int64_t flat_idx_6491;\n    \n    flat_idx_6491 = (sext_i32_i64(local_tid_6479) + (int64_t) 1) *\n        (segscan_group_sizze_6371 * sdiv_up64(implz2080U_6265,\n                                              sext_i32_i64(num_threads_6432))) -\n        (int64_t) 1;\n    \n    int64_t gtid_6375;\n    \n    gtid_6375 = flat_idx_6491;\n    // threads in bound read carries; others get neutral element\n    {\n        if (slt64(gtid_6375, implz2080U_6265)) {\n            ((__local\n              int64_t *) scan_arr_mem_6483)[sext_i32_i64(local_tid_6479)] =\n                ((__global int64_t *) mem_6396)[gtid_6375];\n            ((__local\n              int64_t *) scan_arr_mem_6485)[sext_i32_i64(local_tid_6479)] =\n                ((__global int64_t *) mem_6398)[gtid_6375];\n            ((__local\n              int64_t *) scan_arr_mem_6487)[sext_i32_i64(local_tid_6479)] =\n                ((__global int64_t *) mem_6400)[gtid_6375];\n            ((__local\n              int64_t *) scan_arr_mem_6489)[sext_i32_i64(local_tid_6479)] =\n                ((__global int64_t *) mem_6402)[gtid_6375];\n        } else {\n            ((__local\n              int64_t *) scan_arr_mem_6483)[sext_i32_i64(local_tid_6479)] =\n                (int64_t) 0;\n            ((__local\n              int64_t *) scan_arr_mem_6485)[sext_i32_i64(local_tid_6479)] =\n                (int64_t) 0;\n            ((__local\n              int64_t *) scan_arr_mem_6487)[sext_i32_i64(local_tid_6479)] =\n                (int64_t) 0;\n            ((__local\n              int64_t *) scan_arr_mem_6489)[sext_i32_i64(local_tid_6479)] =\n                (int64_t) 0;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int64_t x_6301;\n    int64_t x_6302;\n    int64_t x_6303;\n    int64_t x_6304;\n    int64_t x_630",
                   "5;\n    int64_t x_6306;\n    int64_t x_6307;\n    int64_t x_6308;\n    int64_t x_6492;\n    int64_t x_6493;\n    int64_t x_6494;\n    int64_t x_6495;\n    int64_t x_6496;\n    int64_t x_6497;\n    int64_t x_6498;\n    int64_t x_6499;\n    bool ltid_in_bounds_6504;\n    \n    ltid_in_bounds_6504 = slt64(sext_i32_i64(local_tid_6479),\n                                stage1_num_groups_6431);\n    \n    int32_t skip_threads_6505;\n    \n    // read input for in-block scan\n    {\n        if (ltid_in_bounds_6504) {\n            x_6305 = ((volatile __local\n                       int64_t *) scan_arr_mem_6483)[sext_i32_i64(local_tid_6479)];\n            x_6306 = ((volatile __local\n                       int64_t *) scan_arr_mem_6485)[sext_i32_i64(local_tid_6479)];\n            x_6307 = ((volatile __local\n                       int64_t *) scan_arr_mem_6487)[sext_i32_i64(local_tid_6479)];\n            x_6308 = ((volatile __local\n                       int64_t *) scan_arr_mem_6489)[sext_i32_i64(local_tid_6479)];\n            if ((local_tid_6479 - squot32(local_tid_6479, 32) * 32) == 0) {\n                x_6301 = x_6305;\n                x_6302 = x_6306;\n                x_6303 = x_6307;\n                x_6304 = x_6308;\n            }\n        }\n    }\n    // in-block scan (hopefully no barriers needed)\n    {\n        skip_threads_6505 = 1;\n        while (slt32(skip_threads_6505, 32)) {\n            if (sle32(skip_threads_6505, local_tid_6479 -\n                      squot32(local_tid_6479, 32) * 32) &&\n                ltid_in_bounds_6504) {\n                // read operands\n                {\n                    x_6301 = ((volatile __local\n                               int64_t *) scan_arr_mem_6483)[sext_i32_i64(local_tid_6479) -\n                                                             sext_i32_i64(skip_threads_6505)];\n                    x_6302 = ((volatile __local\n                               int64_t *) scan_arr_mem_6485)[sext_i32_i64(local_tid_6479) -\n                                                     ",
                   "        sext_i32_i64(skip_threads_6505)];\n                    x_6303 = ((volatile __local\n                               int64_t *) scan_arr_mem_6487)[sext_i32_i64(local_tid_6479) -\n                                                             sext_i32_i64(skip_threads_6505)];\n                    x_6304 = ((volatile __local\n                               int64_t *) scan_arr_mem_6489)[sext_i32_i64(local_tid_6479) -\n                                                             sext_i32_i64(skip_threads_6505)];\n                }\n                // perform operation\n                {\n                    int64_t defunc_1_op_res_6309 = add64(x_6301, x_6305);\n                    int64_t defunc_1_op_res_6310 = add64(x_6302, x_6306);\n                    int64_t defunc_1_op_res_6311 = add64(x_6303, x_6307);\n                    int64_t defunc_1_op_res_6312 = add64(x_6304, x_6308);\n                    \n                    x_6301 = defunc_1_op_res_6309;\n                    x_6302 = defunc_1_op_res_6310;\n                    x_6303 = defunc_1_op_res_6311;\n                    x_6304 = defunc_1_op_res_6312;\n                }\n            }\n            if (sle32(wave_sizze_6481, skip_threads_6505)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            if (sle32(skip_threads_6505, local_tid_6479 -\n                      squot32(local_tid_6479, 32) * 32) &&\n                ltid_in_bounds_6504) {\n                // write result\n                {\n                    ((volatile __local\n                      int64_t *) scan_arr_mem_6483)[sext_i32_i64(local_tid_6479)] =\n                        x_6301;\n                    x_6305 = x_6301;\n                    ((volatile __local\n                      int64_t *) scan_arr_mem_6485)[sext_i32_i64(local_tid_6479)] =\n                        x_6302;\n                    x_6306 = x_6302;\n                    ((volatile __local\n                      int64_t *) scan_arr_mem_6487)[sext_i32_i64(local_tid_6479)] =\n                        x_63",
                   "03;\n                    x_6307 = x_6303;\n                    ((volatile __local\n                      int64_t *) scan_arr_mem_6489)[sext_i32_i64(local_tid_6479)] =\n                        x_6304;\n                    x_6308 = x_6304;\n                }\n            }\n            if (sle32(wave_sizze_6481, skip_threads_6505)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            skip_threads_6505 *= 2;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // last thread of block 'i' writes its result to offset 'i'\n    {\n        if ((local_tid_6479 - squot32(local_tid_6479, 32) * 32) == 31 &&\n            ltid_in_bounds_6504) {\n            ((volatile __local\n              int64_t *) scan_arr_mem_6483)[sext_i32_i64(squot32(local_tid_6479,\n                                                                 32))] = x_6301;\n            ((volatile __local\n              int64_t *) scan_arr_mem_6485)[sext_i32_i64(squot32(local_tid_6479,\n                                                                 32))] = x_6302;\n            ((volatile __local\n              int64_t *) scan_arr_mem_6487)[sext_i32_i64(squot32(local_tid_6479,\n                                                                 32))] = x_6303;\n            ((volatile __local\n              int64_t *) scan_arr_mem_6489)[sext_i32_i64(squot32(local_tid_6479,\n                                                                 32))] = x_6304;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'\n    {\n        int32_t skip_threads_6506;\n        \n        // read input for in-block scan\n        {\n            if (squot32(local_tid_6479, 32) == 0 && ltid_in_bounds_6504) {\n                x_6496 = ((volatile __local\n                           int64_t *) scan_arr_mem_6483)[sext_i32_i64(local_tid_6479)];\n                x_6497 = ((volatile __local\n                           int64_t *) scan_arr_mem_6485)[sext_i32_i64(local_tid_6479)];\n   ",
                   "             x_6498 = ((volatile __local\n                           int64_t *) scan_arr_mem_6487)[sext_i32_i64(local_tid_6479)];\n                x_6499 = ((volatile __local\n                           int64_t *) scan_arr_mem_6489)[sext_i32_i64(local_tid_6479)];\n                if ((local_tid_6479 - squot32(local_tid_6479, 32) * 32) == 0) {\n                    x_6492 = x_6496;\n                    x_6493 = x_6497;\n                    x_6494 = x_6498;\n                    x_6495 = x_6499;\n                }\n            }\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            skip_threads_6506 = 1;\n            while (slt32(skip_threads_6506, 32)) {\n                if (sle32(skip_threads_6506, local_tid_6479 -\n                          squot32(local_tid_6479, 32) * 32) &&\n                    (squot32(local_tid_6479, 32) == 0 && ltid_in_bounds_6504)) {\n                    // read operands\n                    {\n                        x_6492 = ((volatile __local\n                                   int64_t *) scan_arr_mem_6483)[sext_i32_i64(local_tid_6479) -\n                                                                 sext_i32_i64(skip_threads_6506)];\n                        x_6493 = ((volatile __local\n                                   int64_t *) scan_arr_mem_6485)[sext_i32_i64(local_tid_6479) -\n                                                                 sext_i32_i64(skip_threads_6506)];\n                        x_6494 = ((volatile __local\n                                   int64_t *) scan_arr_mem_6487)[sext_i32_i64(local_tid_6479) -\n                                                                 sext_i32_i64(skip_threads_6506)];\n                        x_6495 = ((volatile __local\n                                   int64_t *) scan_arr_mem_6489)[sext_i32_i64(local_tid_6479) -\n                                                                 sext_i32_i64(skip_threads_6506)];\n                    }\n                    // perform operation\n ",
                   "                   {\n                        int64_t defunc_1_op_res_6500 = add64(x_6492, x_6496);\n                        int64_t defunc_1_op_res_6501 = add64(x_6493, x_6497);\n                        int64_t defunc_1_op_res_6502 = add64(x_6494, x_6498);\n                        int64_t defunc_1_op_res_6503 = add64(x_6495, x_6499);\n                        \n                        x_6492 = defunc_1_op_res_6500;\n                        x_6493 = defunc_1_op_res_6501;\n                        x_6494 = defunc_1_op_res_6502;\n                        x_6495 = defunc_1_op_res_6503;\n                    }\n                }\n                if (sle32(wave_sizze_6481, skip_threads_6506)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if (sle32(skip_threads_6506, local_tid_6479 -\n                          squot32(local_tid_6479, 32) * 32) &&\n                    (squot32(local_tid_6479, 32) == 0 && ltid_in_bounds_6504)) {\n                    // write result\n                    {\n                        ((volatile __local\n                          int64_t *) scan_arr_mem_6483)[sext_i32_i64(local_tid_6479)] =\n                            x_6492;\n                        x_6496 = x_6492;\n                        ((volatile __local\n                          int64_t *) scan_arr_mem_6485)[sext_i32_i64(local_tid_6479)] =\n                            x_6493;\n                        x_6497 = x_6493;\n                        ((volatile __local\n                          int64_t *) scan_arr_mem_6487)[sext_i32_i64(local_tid_6479)] =\n                            x_6494;\n                        x_6498 = x_6494;\n                        ((volatile __local\n                          int64_t *) scan_arr_mem_6489)[sext_i32_i64(local_tid_6479)] =\n                            x_6495;\n                        x_6499 = x_6495;\n                    }\n                }\n                if (sle32(wave_sizze_6481, skip_threads_6506)) {\n                    barrier(CLK_LOCAL_MEM_FENCE)",
                   ";\n                }\n                skip_threads_6506 *= 2;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // carry-in for every block except the first\n    {\n        if (!(squot32(local_tid_6479, 32) == 0 || !ltid_in_bounds_6504)) {\n            // read operands\n            {\n                x_6305 = x_6301;\n                x_6306 = x_6302;\n                x_6307 = x_6303;\n                x_6308 = x_6304;\n                x_6301 = ((__local\n                           int64_t *) scan_arr_mem_6483)[sext_i32_i64(squot32(local_tid_6479,\n                                                                              32)) -\n                                                         (int64_t) 1];\n                x_6302 = ((__local\n                           int64_t *) scan_arr_mem_6485)[sext_i32_i64(squot32(local_tid_6479,\n                                                                              32)) -\n                                                         (int64_t) 1];\n                x_6303 = ((__local\n                           int64_t *) scan_arr_mem_6487)[sext_i32_i64(squot32(local_tid_6479,\n                                                                              32)) -\n                                                         (int64_t) 1];\n                x_6304 = ((__local\n                           int64_t *) scan_arr_mem_6489)[sext_i32_i64(squot32(local_tid_6479,\n                                                                              32)) -\n                                                         (int64_t) 1];\n            }\n            // perform operation\n            {\n                int64_t defunc_1_op_res_6309 = add64(x_6301, x_6305);\n                int64_t defunc_1_op_res_6310 = add64(x_6302, x_6306);\n                int64_t defunc_1_op_res_6311 = add64(x_6303, x_6307);\n                int64_t defunc_1_op_res_6312 = add64(x_6304, x_6308);\n                \n                x_6301 = defunc_1_op_res_6309;\n                x_6302 = d",
                   "efunc_1_op_res_6310;\n                x_6303 = defunc_1_op_res_6311;\n                x_6304 = defunc_1_op_res_6312;\n            }\n            // write final result\n            {\n                ((__local\n                  int64_t *) scan_arr_mem_6483)[sext_i32_i64(local_tid_6479)] =\n                    x_6301;\n                ((__local\n                  int64_t *) scan_arr_mem_6485)[sext_i32_i64(local_tid_6479)] =\n                    x_6302;\n                ((__local\n                  int64_t *) scan_arr_mem_6487)[sext_i32_i64(local_tid_6479)] =\n                    x_6303;\n                ((__local\n                  int64_t *) scan_arr_mem_6489)[sext_i32_i64(local_tid_6479)] =\n                    x_6304;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // restore correct values for first block\n    {\n        if (squot32(local_tid_6479, 32) == 0) {\n            ((__local\n              int64_t *) scan_arr_mem_6483)[sext_i32_i64(local_tid_6479)] =\n                x_6305;\n            ((__local\n              int64_t *) scan_arr_mem_6485)[sext_i32_i64(local_tid_6479)] =\n                x_6306;\n            ((__local\n              int64_t *) scan_arr_mem_6487)[sext_i32_i64(local_tid_6479)] =\n                x_6307;\n            ((__local\n              int64_t *) scan_arr_mem_6489)[sext_i32_i64(local_tid_6479)] =\n                x_6308;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // threads in bounds write scanned carries\n    {\n        if (slt64(gtid_6375, implz2080U_6265)) {\n            ((__global int64_t *) mem_6396)[gtid_6375] = ((__local\n                                                           int64_t *) scan_arr_mem_6483)[sext_i32_i64(local_tid_6479)];\n            ((__global int64_t *) mem_6398)[gtid_6375] = ((__local\n                                                           int64_t *) scan_arr_mem_6485)[sext_i32_i64(local_tid_6479)];\n            ((__global int64_t *) mem_6400)[gtid_6375] = ((__local\n                                          ",
                   "                 int64_t *) scan_arr_mem_6487)[sext_i32_i64(local_tid_6479)];\n            ((__global int64_t *) mem_6402)[gtid_6375] = ((__local\n                                                           int64_t *) scan_arr_mem_6489)[sext_i32_i64(local_tid_6479)];\n        }\n    }\n    \n  error_0:\n    return;\n    #undef segscan_group_sizze_6371\n}\n__kernel void mainziscan_stage3_6376(__global int *global_failure,\n                                     int64_t implz2080U_6265,\n                                     int64_t num_groups_6373,\n                                     int32_t num_threads_6432,\n                                     int32_t required_groups_6507, __global\n                                     unsigned char *mem_6396, __global\n                                     unsigned char *mem_6398, __global\n                                     unsigned char *mem_6400, __global\n                                     unsigned char *mem_6402)\n{\n    #define segscan_group_sizze_6371 (mainzisegscan_group_sizze_6370)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_6508;\n    int32_t local_tid_6509;\n    int64_t group_sizze_6512;\n    int32_t wave_sizze_6511;\n    int32_t group_tid_6510;\n    \n    global_tid_6508 = get_global_id(0);\n    local_tid_6509 = get_local_id(0);\n    group_sizze_6512 = get_local_size(0);\n    wave_sizze_6511 = LOCKSTEP_WIDTH;\n    group_tid_6510 = get_group_id(0);\n    \n    int32_t phys_tid_6376;\n    \n    phys_tid_6376 = global_tid_6508;\n    \n    int32_t phys_group_id_6513;\n    \n    phys_group_id_6513 = get_group_id(0);\n    for (int32_t i_6514 = 0; i_6514 < sdiv_up32(required_groups_6507 -\n                                                phys_group_id_6513,\n                                                sext_i64_i32(num_groups_6373));\n         i_6514++) {\n        int32_t virt_group_id_6515 = phys_group_id_6513 + i_6514 *\n               ",
                   " sext_i64_i32(num_groups_6373);\n        int64_t flat_idx_6516 = sext_i32_i64(virt_group_id_6515) *\n                segscan_group_sizze_6371 + sext_i32_i64(local_tid_6509);\n        int64_t gtid_6375 = flat_idx_6516;\n        int64_t orig_group_6517 = squot64(flat_idx_6516,\n                                          segscan_group_sizze_6371 *\n                                          sdiv_up64(implz2080U_6265,\n                                                    sext_i32_i64(num_threads_6432)));\n        int64_t carry_in_flat_idx_6518 = orig_group_6517 *\n                (segscan_group_sizze_6371 * sdiv_up64(implz2080U_6265,\n                                                      sext_i32_i64(num_threads_6432))) -\n                (int64_t) 1;\n        \n        if (slt64(gtid_6375, implz2080U_6265)) {\n            if (!(orig_group_6517 == (int64_t) 0 || flat_idx_6516 ==\n                  (orig_group_6517 + (int64_t) 1) * (segscan_group_sizze_6371 *\n                                                     sdiv_up64(implz2080U_6265,\n                                                               sext_i32_i64(num_threads_6432))) -\n                  (int64_t) 1)) {\n                int64_t x_6301;\n                int64_t x_6302;\n                int64_t x_6303;\n                int64_t x_6304;\n                int64_t x_6305;\n                int64_t x_6306;\n                int64_t x_6307;\n                int64_t x_6308;\n                \n                x_6301 = ((__global\n                           int64_t *) mem_6396)[carry_in_flat_idx_6518];\n                x_6302 = ((__global\n                           int64_t *) mem_6398)[carry_in_flat_idx_6518];\n                x_6303 = ((__global\n                           int64_t *) mem_6400)[carry_in_flat_idx_6518];\n                x_6304 = ((__global\n                           int64_t *) mem_6402)[carry_in_flat_idx_6518];\n                x_6305 = ((__global int64_t *) mem_6396)[gtid_6375];\n                x_6306 = ((__global int64_t *) mem_6398)[gt",
                   "id_6375];\n                x_6307 = ((__global int64_t *) mem_6400)[gtid_6375];\n                x_6308 = ((__global int64_t *) mem_6402)[gtid_6375];\n                \n                int64_t defunc_1_op_res_6309;\n                \n                defunc_1_op_res_6309 = add64(x_6301, x_6305);\n                \n                int64_t defunc_1_op_res_6310 = add64(x_6302, x_6306);\n                int64_t defunc_1_op_res_6311 = add64(x_6303, x_6307);\n                int64_t defunc_1_op_res_6312 = add64(x_6304, x_6308);\n                \n                x_6301 = defunc_1_op_res_6309;\n                x_6302 = defunc_1_op_res_6310;\n                x_6303 = defunc_1_op_res_6311;\n                x_6304 = defunc_1_op_res_6312;\n                ((__global int64_t *) mem_6396)[gtid_6375] = x_6301;\n                ((__global int64_t *) mem_6398)[gtid_6375] = x_6302;\n                ((__global int64_t *) mem_6400)[gtid_6375] = x_6303;\n                ((__global int64_t *) mem_6402)[gtid_6375] = x_6304;\n            }\n        }\n        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);\n    }\n    \n  error_0:\n    return;\n    #undef segscan_group_sizze_6371\n}\n__kernel void mainzisegmap_6378(__global int *global_failure,\n                                int64_t implz2080U_6265, int64_t last_res_6342,\n                                int64_t last_res_6343, int64_t last_res_6344,\n                                __global unsigned char *mem_param_6391, __global\n                                unsigned char *mem_6396, __global\n                                unsigned char *mem_6398, __global\n                                unsigned char *mem_6400, __global\n                                unsigned char *mem_6402, __global\n                                unsigned char *mem_6404, __global\n                                unsigned char *mem_6406)\n{\n    #define segmap_group_sizze_6381 (mainzisegmap_group_sizze_6380)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_di",
                   "m2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_6519;\n    int32_t local_tid_6520;\n    int64_t group_sizze_6523;\n    int32_t wave_sizze_6522;\n    int32_t group_tid_6521;\n    \n    global_tid_6519 = get_global_id(0);\n    local_tid_6520 = get_local_id(0);\n    group_sizze_6523 = get_local_size(0);\n    wave_sizze_6522 = LOCKSTEP_WIDTH;\n    group_tid_6521 = get_group_id(0);\n    \n    int32_t phys_tid_6378;\n    \n    phys_tid_6378 = global_tid_6519;\n    \n    int64_t write_i_6377;\n    \n    write_i_6377 = sext_i32_i64(group_tid_6521) * segmap_group_sizze_6381 +\n        sext_i32_i64(local_tid_6520);\n    if (slt64(write_i_6377, implz2080U_6265)) {\n        int32_t x_6347 = ((__global int32_t *) mem_6404)[write_i_6377];\n        int32_t write_value_6352 = ((__global\n                                     int32_t *) mem_param_6391)[write_i_6377];\n        bool match_lit_6353 = 0 == x_6347;\n        int64_t defunc_1_f_res_6354;\n        \n        if (match_lit_6353) {\n            int64_t x_6348 = ((__global int64_t *) mem_6396)[write_i_6377];\n            int64_t case_res_6355 = sub64(x_6348, (int64_t) 1);\n            \n            defunc_1_f_res_6354 = case_res_6355;\n        } else {\n            int64_t x_6349 = ((__global int64_t *) mem_6398)[write_i_6377];\n            int64_t x_6350 = ((__global int64_t *) mem_6400)[write_i_6377];\n            int64_t x_6351 = ((__global int64_t *) mem_6402)[write_i_6377];\n            bool match_lit_6356 = 1 == x_6347;\n            int64_t x_6357;\n            \n            if (match_lit_6356) {\n                int64_t x_6358 = add64(last_res_6342, x_6349);\n                int64_t case_res_6359 = sub64(x_6358, (int64_t) 1);\n                \n                x_6357 = case_res_6359;\n            } else {\n                bool match_lit_6360 = 2 == x_6347;\n                int64_t x_6361;\n                \n                if (match_lit_6360) {\n                    int64_t x_6362 = add64(last_res_6342, last_res_6343);\n             ",
                   "       int64_t x_6363 = add64(x_6350, x_6362);\n                    int64_t case_res_6364 = sub64(x_6363, (int64_t) 1);\n                    \n                    x_6361 = case_res_6364;\n                } else {\n                    int64_t x_6365 = add64(last_res_6342, last_res_6343);\n                    int64_t x_6366 = add64(last_res_6344, x_6365);\n                    int64_t x_6367 = add64(x_6351, x_6366);\n                    int64_t defunc_1_f_res_res_6368 = sub64(x_6367,\n                                                            (int64_t) 1);\n                    \n                    x_6361 = defunc_1_f_res_res_6368;\n                }\n                x_6357 = x_6361;\n            }\n            defunc_1_f_res_6354 = x_6357;\n        }\n        if (sle64((int64_t) 0, defunc_1_f_res_6354) &&\n            slt64(defunc_1_f_res_6354, implz2080U_6265)) {\n            ((__global int32_t *) mem_6406)[defunc_1_f_res_6354] =\n                write_value_6352;\n        }\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_6381\n}\n",
                   NULL};
static const char *size_names[] = {"main.segmap_group_size_6380",
                                   "main.segscan_group_size_6370",
                                   "main.segscan_num_groups_6372"};
static const char *size_vars[] = {"mainzisegmap_group_sizze_6380",
                                  "mainzisegscan_group_sizze_6370",
                                  "mainzisegscan_num_groups_6372"};
static const char *size_classes[] = {"group_size", "group_size", "num_groups"};
int futhark_get_num_sizes(void)
{
    return 3;
}
const char *futhark_get_size_name(int i)
{
    return size_names[i];
}
const char *futhark_get_size_class(int i)
{
    return size_classes[i];
}
struct sizes {
    int64_t mainzisegmap_group_sizze_6380;
    int64_t mainzisegscan_group_sizze_6370;
    int64_t mainzisegscan_num_groups_6372;
} ;
struct futhark_context_config {
    struct opencl_config opencl;
    int64_t sizes[3];
    int num_build_opts;
    const char **build_opts;
} ;
struct futhark_context_config *futhark_context_config_new(void)
{
    struct futhark_context_config *cfg =
                                  (struct futhark_context_config *) malloc(sizeof(struct futhark_context_config));
    
    if (cfg == NULL)
        return NULL;
    cfg->num_build_opts = 0;
    cfg->build_opts = (const char **) malloc(sizeof(const char *));
    cfg->build_opts[0] = NULL;
    cfg->sizes[0] = 0;
    cfg->sizes[1] = 0;
    cfg->sizes[2] = 0;
    opencl_config_init(&cfg->opencl, 3, size_names, size_vars, cfg->sizes,
                       size_classes);
    return cfg;
}
void futhark_context_config_free(struct futhark_context_config *cfg)
{
    free(cfg->build_opts);
    free(cfg);
}
void futhark_context_config_add_build_option(struct futhark_context_config *cfg,
                                             const char *opt)
{
    cfg->build_opts[cfg->num_build_opts] = opt;
    cfg->num_build_opts++;
    cfg->build_opts = (const char **) realloc(cfg->build_opts,
                                              (cfg->num_build_opts + 1) *
                                              sizeof(const char *));
    cfg->build_opts[cfg->num_build_opts] = NULL;
}
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag)
{
    cfg->opencl.profiling = cfg->opencl.logging = cfg->opencl.debugging = flag;
}
void futhark_context_config_set_profiling(struct futhark_context_config *cfg,
                                          int flag)
{
    cfg->opencl.profiling = flag;
}
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag)
{
    cfg->opencl.logging = flag;
}
void futhark_context_config_set_device(struct futhark_context_config *cfg, const
                                       char *s)
{
    set_preferred_device(&cfg->opencl, s);
}
void futhark_context_config_set_platform(struct futhark_context_config *cfg,
                                         const char *s)
{
    set_preferred_platform(&cfg->opencl, s);
}
void futhark_context_config_select_device_interactively(struct futhark_context_config *cfg)
{
    select_device_interactively(&cfg->opencl);
}
void futhark_context_config_list_devices(struct futhark_context_config *cfg)
{
    (void) cfg;
    list_devices();
}
void futhark_context_config_dump_program_to(struct futhark_context_config *cfg,
                                            const char *path)
{
    cfg->opencl.dump_program_to = path;
}
void futhark_context_config_load_program_from(struct futhark_context_config *cfg,
                                              const char *path)
{
    cfg->opencl.load_program_from = path;
}
void futhark_context_config_dump_binary_to(struct futhark_context_config *cfg,
                                           const char *path)
{
    cfg->opencl.dump_binary_to = path;
}
void futhark_context_config_load_binary_from(struct futhark_context_config *cfg,
                                             const char *path)
{
    cfg->opencl.load_binary_from = path;
}
void futhark_context_config_set_default_group_size(struct futhark_context_config *cfg,
                                                   int size)
{
    cfg->opencl.default_group_size = size;
    cfg->opencl.default_group_size_changed = 1;
}
void futhark_context_config_set_default_num_groups(struct futhark_context_config *cfg,
                                                   int num)
{
    cfg->opencl.default_num_groups = num;
}
void futhark_context_config_set_default_tile_size(struct futhark_context_config *cfg,
                                                  int size)
{
    cfg->opencl.default_tile_size = size;
    cfg->opencl.default_tile_size_changed = 1;
}
void futhark_context_config_set_default_threshold(struct futhark_context_config *cfg,
                                                  int size)
{
    cfg->opencl.default_threshold = size;
}
int futhark_context_config_set_size(struct futhark_context_config *cfg, const
                                    char *size_name, size_t size_value)
{
    for (int i = 0; i < 3; i++) {
        if (strcmp(size_name, size_names[i]) == 0) {
            cfg->sizes[i] = size_value;
            return 0;
        }
    }
    if (strcmp(size_name, "default_group_size") == 0) {
        cfg->opencl.default_group_size = size_value;
        return 0;
    }
    if (strcmp(size_name, "default_num_groups") == 0) {
        cfg->opencl.default_num_groups = size_value;
        return 0;
    }
    if (strcmp(size_name, "default_threshold") == 0) {
        cfg->opencl.default_threshold = size_value;
        return 0;
    }
    if (strcmp(size_name, "default_tile_size") == 0) {
        cfg->opencl.default_tile_size = size_value;
        return 0;
    }
    return 1;
}
struct futhark_context {
    int detail_memory;
    int debugging;
    int profiling;
    int profiling_paused;
    int logging;
    lock_t lock;
    char *error;
    FILE *log;
    int64_t peak_mem_usage_device;
    int64_t cur_mem_usage_device;
    int64_t peak_mem_usage_default;
    int64_t cur_mem_usage_default;
    struct {
        int dummy;
    } constants;
    int total_runs;
    long total_runtime;
    cl_kernel mainziscan_stage1_6376;
    cl_kernel mainziscan_stage2_6376;
    cl_kernel mainziscan_stage3_6376;
    cl_kernel mainzisegmap_6378;
    int64_t copy_dev_to_dev_total_runtime;
    int copy_dev_to_dev_runs;
    int64_t copy_dev_to_host_total_runtime;
    int copy_dev_to_host_runs;
    int64_t copy_host_to_dev_total_runtime;
    int copy_host_to_dev_runs;
    int64_t copy_scalar_to_dev_total_runtime;
    int copy_scalar_to_dev_runs;
    int64_t copy_scalar_from_dev_total_runtime;
    int copy_scalar_from_dev_runs;
    int64_t mainziscan_stage1_6376_total_runtime;
    int mainziscan_stage1_6376_runs;
    int64_t mainziscan_stage2_6376_total_runtime;
    int mainziscan_stage2_6376_runs;
    int64_t mainziscan_stage3_6376_total_runtime;
    int mainziscan_stage3_6376_runs;
    int64_t mainzisegmap_6378_total_runtime;
    int mainzisegmap_6378_runs;
    cl_mem global_failure;
    cl_mem global_failure_args;
    struct opencl_context opencl;
    struct sizes sizes;
    cl_int failure_is_an_option;
} ;
void post_opencl_setup(struct opencl_context *ctx,
                       struct opencl_device_option *option)
{
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name,
                                            "NVIDIA CUDA") != NULL) &&
        (option->device_type & CL_DEVICE_TYPE_GPU) == CL_DEVICE_TYPE_GPU) {
        ctx->lockstep_width = 32;
    }
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name,
                                            "AMD Accelerated Parallel Processing") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_GPU) ==
        CL_DEVICE_TYPE_GPU) {
        ctx->lockstep_width = 32;
    }
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_GPU) ==
        CL_DEVICE_TYPE_GPU) {
        ctx->lockstep_width = 1;
    }
    if ((ctx->cfg.default_num_groups == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        (option->device_type & CL_DEVICE_TYPE_GPU) == CL_DEVICE_TYPE_GPU) {
        size_t MAX_COMPUTE_UNITS_val = 0;
        
        clGetDeviceInfo(ctx->device, CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(MAX_COMPUTE_UNITS_val), &MAX_COMPUTE_UNITS_val,
                        NULL);
        ctx->cfg.default_num_groups = 4 * MAX_COMPUTE_UNITS_val;
    }
    if ((ctx->cfg.default_group_size == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        (option->device_type & CL_DEVICE_TYPE_GPU) == CL_DEVICE_TYPE_GPU) {
        ctx->cfg.default_group_size = 256;
    }
    if ((ctx->cfg.default_tile_size == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_GPU) ==
        CL_DEVICE_TYPE_GPU) {
        ctx->cfg.default_tile_size = 32;
    }
    if ((ctx->cfg.default_threshold == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_GPU) ==
        CL_DEVICE_TYPE_GPU) {
        ctx->cfg.default_threshold = 32768;
    }
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_CPU) ==
        CL_DEVICE_TYPE_CPU) {
        ctx->lockstep_width = 1;
    }
    if ((ctx->cfg.default_num_groups == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        (option->device_type & CL_DEVICE_TYPE_CPU) == CL_DEVICE_TYPE_CPU) {
        size_t MAX_COMPUTE_UNITS_val = 0;
        
        clGetDeviceInfo(ctx->device, CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(MAX_COMPUTE_UNITS_val), &MAX_COMPUTE_UNITS_val,
                        NULL);
        ctx->cfg.default_num_groups = MAX_COMPUTE_UNITS_val;
    }
    if ((ctx->cfg.default_group_size == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        (option->device_type & CL_DEVICE_TYPE_CPU) == CL_DEVICE_TYPE_CPU) {
        ctx->cfg.default_group_size = 32;
    }
    if ((ctx->cfg.default_tile_size == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_CPU) ==
        CL_DEVICE_TYPE_CPU) {
        ctx->cfg.default_tile_size = 4;
    }
    if ((ctx->cfg.default_threshold == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_CPU) ==
        CL_DEVICE_TYPE_CPU) {
        size_t MAX_COMPUTE_UNITS_val = 0;
        
        clGetDeviceInfo(ctx->device, CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(MAX_COMPUTE_UNITS_val), &MAX_COMPUTE_UNITS_val,
                        NULL);
        ctx->cfg.default_threshold = MAX_COMPUTE_UNITS_val;
    }
}
static void init_context_early(struct futhark_context_config *cfg,
                               struct futhark_context *ctx)
{
    ctx->opencl.cfg = cfg->opencl;
    ctx->detail_memory = cfg->opencl.debugging;
    ctx->debugging = cfg->opencl.debugging;
    ctx->profiling = cfg->opencl.profiling;
    ctx->profiling_paused = 0;
    ctx->logging = cfg->opencl.logging;
    ctx->error = NULL;
    ctx->log = stderr;
    ctx->opencl.profiling_records_capacity = 200;
    ctx->opencl.profiling_records_used = 0;
    ctx->opencl.profiling_records =
        malloc(ctx->opencl.profiling_records_capacity *
        sizeof(struct profiling_record));
    create_lock(&ctx->lock);
    ctx->failure_is_an_option = 0;
    ctx->peak_mem_usage_device = 0;
    ctx->cur_mem_usage_device = 0;
    ctx->peak_mem_usage_default = 0;
    ctx->cur_mem_usage_default = 0;
    ctx->total_runs = 0;
    ctx->total_runtime = 0;
    ctx->copy_dev_to_dev_total_runtime = 0;
    ctx->copy_dev_to_dev_runs = 0;
    ctx->copy_dev_to_host_total_runtime = 0;
    ctx->copy_dev_to_host_runs = 0;
    ctx->copy_host_to_dev_total_runtime = 0;
    ctx->copy_host_to_dev_runs = 0;
    ctx->copy_scalar_to_dev_total_runtime = 0;
    ctx->copy_scalar_to_dev_runs = 0;
    ctx->copy_scalar_from_dev_total_runtime = 0;
    ctx->copy_scalar_from_dev_runs = 0;
    ctx->mainziscan_stage1_6376_total_runtime = 0;
    ctx->mainziscan_stage1_6376_runs = 0;
    ctx->mainziscan_stage2_6376_total_runtime = 0;
    ctx->mainziscan_stage2_6376_runs = 0;
    ctx->mainziscan_stage3_6376_total_runtime = 0;
    ctx->mainziscan_stage3_6376_runs = 0;
    ctx->mainzisegmap_6378_total_runtime = 0;
    ctx->mainzisegmap_6378_runs = 0;
}
static int init_context_late(struct futhark_context_config *cfg,
                             struct futhark_context *ctx, cl_program prog)
{
    cl_int error;
    cl_int no_error = -1;
    
    ctx->global_failure = clCreateBuffer(ctx->opencl.ctx, CL_MEM_READ_WRITE |
                                         CL_MEM_COPY_HOST_PTR, sizeof(cl_int),
                                         &no_error, &error);
    OPENCL_SUCCEED_OR_RETURN(error);
    // The +1 is to avoid zero-byte allocations.
    ctx->global_failure_args = clCreateBuffer(ctx->opencl.ctx,
                                              CL_MEM_READ_WRITE,
                                              sizeof(int64_t) * (0 + 1), NULL,
                                              &error);
    OPENCL_SUCCEED_OR_RETURN(error);
    {
        ctx->mainziscan_stage1_6376 = clCreateKernel(prog,
                                                     "mainziscan_stage1_6376",
                                                     &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->mainziscan_stage1_6376, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        if (ctx->debugging)
            fprintf(ctx->log, "Created kernel %s.\n", "main.scan_stage1_6376");
    }
    {
        ctx->mainziscan_stage2_6376 = clCreateKernel(prog,
                                                     "mainziscan_stage2_6376",
                                                     &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->mainziscan_stage2_6376, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        if (ctx->debugging)
            fprintf(ctx->log, "Created kernel %s.\n", "main.scan_stage2_6376");
    }
    {
        ctx->mainziscan_stage3_6376 = clCreateKernel(prog,
                                                     "mainziscan_stage3_6376",
                                                     &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->mainziscan_stage3_6376, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        if (ctx->debugging)
            fprintf(ctx->log, "Created kernel %s.\n", "main.scan_stage3_6376");
    }
    {
        ctx->mainzisegmap_6378 = clCreateKernel(prog, "mainzisegmap_6378",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->mainzisegmap_6378, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        if (ctx->debugging)
            fprintf(ctx->log, "Created kernel %s.\n", "main.segmap_6378");
    }
    ctx->sizes.mainzisegmap_group_sizze_6380 = cfg->sizes[0];
    ctx->sizes.mainzisegscan_group_sizze_6370 = cfg->sizes[1];
    ctx->sizes.mainzisegscan_num_groups_6372 = cfg->sizes[2];
    init_constants(ctx);
    // Clear the free list of any deallocations that occurred while initialising constants.
    OPENCL_SUCCEED_OR_RETURN(opencl_free_all(&ctx->opencl));
    // The program will be properly freed after all the kernels have also been freed.
    OPENCL_SUCCEED_OR_RETURN(clReleaseProgram(prog));
    return futhark_context_sync(ctx);
}
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg)
{
    struct futhark_context *ctx =
                           (struct futhark_context *) malloc(sizeof(struct futhark_context));
    
    if (ctx == NULL)
        return NULL;
    
    int required_types = 0;
    
    init_context_early(cfg, ctx);
    
    cl_program prog = setup_opencl(&ctx->opencl, opencl_program, required_types,
                                   cfg->build_opts);
    
    init_context_late(cfg, ctx, prog);
    return ctx;
}
struct futhark_context *futhark_context_new_with_command_queue(struct futhark_context_config *cfg,
                                                               cl_command_queue queue)
{
    struct futhark_context *ctx =
                           (struct futhark_context *) malloc(sizeof(struct futhark_context));
    
    if (ctx == NULL)
        return NULL;
    
    int required_types = 0;
    
    init_context_early(cfg, ctx);
    
    cl_program prog = setup_opencl_with_command_queue(&ctx->opencl, queue,
                                                      opencl_program,
                                                      required_types,
                                                      cfg->build_opts);
    
    init_context_late(cfg, ctx, prog);
    return ctx;
}
void futhark_context_free(struct futhark_context *ctx)
{
    free_constants(ctx);
    free_lock(&ctx->lock);
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->mainziscan_stage1_6376));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->mainziscan_stage2_6376));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->mainziscan_stage3_6376));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->mainzisegmap_6378));
    teardown_opencl(&ctx->opencl);
    free(ctx);
}
int futhark_context_sync(struct futhark_context *ctx)
{
    cl_int failure_idx = -1;
    
    if (ctx->failure_is_an_option) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                     ctx->global_failure,
                                                     CL_FALSE, 0,
                                                     sizeof(cl_int),
                                                     &failure_idx, 0, NULL,
                                                     ctx->profiling_paused ||
                                                     !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                               &ctx->copy_scalar_from_dev_runs,
                                                                                               &ctx->copy_scalar_from_dev_total_runtime)));
        ctx->failure_is_an_option = 0;
    }
    OPENCL_SUCCEED_OR_RETURN(clFinish(ctx->opencl.queue));
    if (failure_idx >= 0) {
        cl_int no_failure = -1;
        
        OPENCL_SUCCEED_OR_RETURN(clEnqueueWriteBuffer(ctx->opencl.queue,
                                                      ctx->global_failure,
                                                      CL_TRUE, 0,
                                                      sizeof(cl_int),
                                                      &no_failure, 0, NULL,
                                                      NULL));
        
        int64_t args[0 + 1];
        
        OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                     ctx->global_failure_args,
                                                     CL_TRUE, 0, sizeof(args),
                                                     &args, 0, NULL,
                                                     ctx->profiling_paused ||
                                                     !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                               &ctx->copy_dev_to_host_runs,
                                                                                               &ctx->copy_dev_to_host_total_runtime)));
        switch (failure_idx) { }
        return 1;
    }
    return 0;
}
cl_command_queue futhark_context_get_command_queue(struct futhark_context *ctx)
{
    return ctx->opencl.queue;
}
static int memblock_unref_device(struct futhark_context *ctx,
                                 struct memblock_device *block, const
                                 char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(ctx->log,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "space 'device'", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_device -= block->size;
            OPENCL_SUCCEED_OR_RETURN(opencl_free(&ctx->opencl, block->mem,
                                                 desc));
            free(block->references);
            if (ctx->detail_memory)
                fprintf(ctx->log,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_device);
        }
        block->references = NULL;
    }
    return 0;
}
static int memblock_alloc_device(struct futhark_context *ctx,
                                 struct memblock_device *block, int64_t size,
                                 const char *desc)
{
    if (size < 0)
        futhark_panic(1,
                      "Negative allocation of %lld bytes attempted for %s in %s.\n",
                      (long long) size, desc, "space 'device'",
                      ctx->cur_mem_usage_device);
    
    int ret = memblock_unref_device(ctx, block, desc);
    
    ctx->cur_mem_usage_device += size;
    if (ctx->detail_memory)
        fprintf(ctx->log,
                "Allocating %lld bytes for %s in %s (then allocated: %lld bytes)",
                (long long) size, desc, "space 'device'",
                (long long) ctx->cur_mem_usage_device);
    if (ctx->cur_mem_usage_device > ctx->peak_mem_usage_device) {
        ctx->peak_mem_usage_device = ctx->cur_mem_usage_device;
        if (ctx->detail_memory)
            fprintf(ctx->log, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(ctx->log, ".\n");
    OPENCL_SUCCEED_OR_RETURN(opencl_alloc(&ctx->opencl, size, desc,
                                          &block->mem));
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    return ret;
}
static int memblock_set_device(struct futhark_context *ctx,
                               struct memblock_device *lhs,
                               struct memblock_device *rhs, const
                               char *lhs_desc)
{
    int ret = memblock_unref_device(ctx, lhs, lhs_desc);
    
    if (rhs->references != NULL)
        (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
static int memblock_unref(struct futhark_context *ctx, struct memblock *block,
                          const char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(ctx->log,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "default space", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_default -= block->size;
            free(block->mem);
            free(block->references);
            if (ctx->detail_memory)
                fprintf(ctx->log,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_default);
        }
        block->references = NULL;
    }
    return 0;
}
static int memblock_alloc(struct futhark_context *ctx, struct memblock *block,
                          int64_t size, const char *desc)
{
    if (size < 0)
        futhark_panic(1,
                      "Negative allocation of %lld bytes attempted for %s in %s.\n",
                      (long long) size, desc, "default space",
                      ctx->cur_mem_usage_default);
    
    int ret = memblock_unref(ctx, block, desc);
    
    ctx->cur_mem_usage_default += size;
    if (ctx->detail_memory)
        fprintf(ctx->log,
                "Allocating %lld bytes for %s in %s (then allocated: %lld bytes)",
                (long long) size, desc, "default space",
                (long long) ctx->cur_mem_usage_default);
    if (ctx->cur_mem_usage_default > ctx->peak_mem_usage_default) {
        ctx->peak_mem_usage_default = ctx->cur_mem_usage_default;
        if (ctx->detail_memory)
            fprintf(ctx->log, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(ctx->log, ".\n");
    block->mem = (char *) malloc(size);
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    return ret;
}
static int memblock_set(struct futhark_context *ctx, struct memblock *lhs,
                        struct memblock *rhs, const char *lhs_desc)
{
    int ret = memblock_unref(ctx, lhs, lhs_desc);
    
    if (rhs->references != NULL)
        (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
char *futhark_context_report(struct futhark_context *ctx)
{
    struct str_builder builder;
    
    str_builder_init(&builder);
    if (ctx->detail_memory || ctx->profiling || ctx->logging) {
        str_builder(&builder,
                    "Peak memory usage for space 'device': %lld bytes.\n",
                    (long long) ctx->peak_mem_usage_device);
        { }
    }
    if (ctx->profiling) {
        OPENCL_SUCCEED_FATAL(opencl_tally_profiling_records(&ctx->opencl));
        str_builder(&builder,
                    "copy_dev_to_dev       ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->copy_dev_to_dev_runs,
                    (long) ctx->copy_dev_to_dev_total_runtime /
                    (ctx->copy_dev_to_dev_runs !=
                     0 ? ctx->copy_dev_to_dev_runs : 1),
                    (long) ctx->copy_dev_to_dev_total_runtime);
        ctx->total_runtime += ctx->copy_dev_to_dev_total_runtime;
        ctx->total_runs += ctx->copy_dev_to_dev_runs;
        str_builder(&builder,
                    "copy_dev_to_host      ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->copy_dev_to_host_runs,
                    (long) ctx->copy_dev_to_host_total_runtime /
                    (ctx->copy_dev_to_host_runs !=
                     0 ? ctx->copy_dev_to_host_runs : 1),
                    (long) ctx->copy_dev_to_host_total_runtime);
        ctx->total_runtime += ctx->copy_dev_to_host_total_runtime;
        ctx->total_runs += ctx->copy_dev_to_host_runs;
        str_builder(&builder,
                    "copy_host_to_dev      ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->copy_host_to_dev_runs,
                    (long) ctx->copy_host_to_dev_total_runtime /
                    (ctx->copy_host_to_dev_runs !=
                     0 ? ctx->copy_host_to_dev_runs : 1),
                    (long) ctx->copy_host_to_dev_total_runtime);
        ctx->total_runtime += ctx->copy_host_to_dev_total_runtime;
        ctx->total_runs += ctx->copy_host_to_dev_runs;
        str_builder(&builder,
                    "copy_scalar_to_dev    ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->copy_scalar_to_dev_runs,
                    (long) ctx->copy_scalar_to_dev_total_runtime /
                    (ctx->copy_scalar_to_dev_runs !=
                     0 ? ctx->copy_scalar_to_dev_runs : 1),
                    (long) ctx->copy_scalar_to_dev_total_runtime);
        ctx->total_runtime += ctx->copy_scalar_to_dev_total_runtime;
        ctx->total_runs += ctx->copy_scalar_to_dev_runs;
        str_builder(&builder,
                    "copy_scalar_from_dev  ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->copy_scalar_from_dev_runs,
                    (long) ctx->copy_scalar_from_dev_total_runtime /
                    (ctx->copy_scalar_from_dev_runs !=
                     0 ? ctx->copy_scalar_from_dev_runs : 1),
                    (long) ctx->copy_scalar_from_dev_total_runtime);
        ctx->total_runtime += ctx->copy_scalar_from_dev_total_runtime;
        ctx->total_runs += ctx->copy_scalar_from_dev_runs;
        str_builder(&builder,
                    "main.scan_stage1_6376 ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->mainziscan_stage1_6376_runs,
                    (long) ctx->mainziscan_stage1_6376_total_runtime /
                    (ctx->mainziscan_stage1_6376_runs !=
                     0 ? ctx->mainziscan_stage1_6376_runs : 1),
                    (long) ctx->mainziscan_stage1_6376_total_runtime);
        ctx->total_runtime += ctx->mainziscan_stage1_6376_total_runtime;
        ctx->total_runs += ctx->mainziscan_stage1_6376_runs;
        str_builder(&builder,
                    "main.scan_stage2_6376 ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->mainziscan_stage2_6376_runs,
                    (long) ctx->mainziscan_stage2_6376_total_runtime /
                    (ctx->mainziscan_stage2_6376_runs !=
                     0 ? ctx->mainziscan_stage2_6376_runs : 1),
                    (long) ctx->mainziscan_stage2_6376_total_runtime);
        ctx->total_runtime += ctx->mainziscan_stage2_6376_total_runtime;
        ctx->total_runs += ctx->mainziscan_stage2_6376_runs;
        str_builder(&builder,
                    "main.scan_stage3_6376 ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->mainziscan_stage3_6376_runs,
                    (long) ctx->mainziscan_stage3_6376_total_runtime /
                    (ctx->mainziscan_stage3_6376_runs !=
                     0 ? ctx->mainziscan_stage3_6376_runs : 1),
                    (long) ctx->mainziscan_stage3_6376_total_runtime);
        ctx->total_runtime += ctx->mainziscan_stage3_6376_total_runtime;
        ctx->total_runs += ctx->mainziscan_stage3_6376_runs;
        str_builder(&builder,
                    "main.segmap_6378      ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->mainzisegmap_6378_runs,
                    (long) ctx->mainzisegmap_6378_total_runtime /
                    (ctx->mainzisegmap_6378_runs !=
                     0 ? ctx->mainzisegmap_6378_runs : 1),
                    (long) ctx->mainzisegmap_6378_total_runtime);
        ctx->total_runtime += ctx->mainzisegmap_6378_total_runtime;
        ctx->total_runs += ctx->mainzisegmap_6378_runs;
        str_builder(&builder, "%d operations with cumulative runtime: %6ldus\n",
                    ctx->total_runs, ctx->total_runtime);
    }
    return builder.str;
}
char *futhark_context_get_error(struct futhark_context *ctx)
{
    char *error = ctx->error;
    
    ctx->error = NULL;
    return error;
}
void futhark_context_set_logging_file(struct futhark_context *ctx, FILE *f)
{
    ctx->log = f;
}
void futhark_context_pause_profiling(struct futhark_context *ctx)
{
    ctx->profiling_paused = 1;
}
void futhark_context_unpause_profiling(struct futhark_context *ctx)
{
    ctx->profiling_paused = 0;
}
int futhark_context_clear_caches(struct futhark_context *ctx)
{
    lock_lock(&ctx->lock);
    ctx->peak_mem_usage_device = 0;
    ctx->peak_mem_usage_default = 0;
    if (ctx->error == NULL)
        ctx->error = OPENCL_SUCCEED_NONFATAL(opencl_free_all(&ctx->opencl));
    lock_unlock(&ctx->lock);
    return ctx->error != NULL;
}
static int futrts_main(struct futhark_context *ctx,
                       struct memblock_device *out_mem_p_6524,
                       struct memblock_device n_mem_6386,
                       int64_t implz2080U_6265);
static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    return 0;
}
static int futrts_main(struct futhark_context *ctx,
                       struct memblock_device *out_mem_p_6524,
                       struct memblock_device n_mem_6386,
                       int64_t implz2080U_6265)
{
    (void) ctx;
    
    int err = 0;
    struct memblock_device out_mem_6426;
    
    out_mem_6426.references = NULL;
    
    bool cond_6267 = implz2080U_6265 == (int64_t) 0;
    int32_t iters_6268;
    
    if (cond_6267) {
        iters_6268 = 0;
    } else {
        iters_6268 = 16;
    }
    
    bool loop_nonempty_6269 = slt32(0, iters_6268);
    int64_t i_6270 = sub64(implz2080U_6265, (int64_t) 1);
    bool x_6271 = sle64((int64_t) 0, i_6270);
    bool y_6272 = slt64(i_6270, implz2080U_6265);
    bool bounds_check_6273 = x_6271 && y_6272;
    bool loop_not_taken_6274 = !loop_nonempty_6269;
    bool protect_assert_disj_6275 = bounds_check_6273 || loop_not_taken_6274;
    bool index_certs_6276;
    
    if (!protect_assert_disj_6275) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Index [", i_6270,
                               "] out of bounds for array of shape [",
                               implz2080U_6265, "].",
                               "-> #0  /prelude/array.fut:18:29-34\n   #1  radix_sort.fut:25:24-35\n   #2  radix_sort.fut:53:31-62\n   #3  radix_sort.fut:79:6-36\n   #4  radix_sort_futhark.fut:7:5-45\n   #5  radix_sort_futhark.fut:6:1-7:45\n");
        if (memblock_unref_device(ctx, &out_mem_6426, "out_mem_6426") != 0)
            return 1;
        return 1;
    }
    
    int64_t segscan_group_sizze_6371;
    
    segscan_group_sizze_6371 = ctx->sizes.mainzisegscan_group_sizze_6370;
    
    int64_t num_groups_6373;
    int32_t max_num_groups_6427;
    
    max_num_groups_6427 = ctx->sizes.mainzisegscan_num_groups_6372;
    num_groups_6373 = sext_i64_i32(smax64((int64_t) 1,
                                          smin64(sdiv_up64(implz2080U_6265,
                                                           segscan_group_sizze_6371),
                                                 sext_i32_i64(max_num_groups_6427))));
    
    int64_t segmap_group_sizze_6381;
    
    segmap_group_sizze_6381 = ctx->sizes.mainzisegmap_group_sizze_6380;
    
    int64_t segmap_usable_groups_6382 = sdiv_up_safe64(implz2080U_6265,
                                                       segmap_group_sizze_6381);
    int64_t bytes_6395 = (int64_t) 8 * implz2080U_6265;
    struct memblock_device mem_6396;
    
    mem_6396.references = NULL;
    if (memblock_alloc_device(ctx, &mem_6396, bytes_6395, "mem_6396")) {
        err = 1;
        goto cleanup;
    }
    
    struct memblock_device mem_6398;
    
    mem_6398.references = NULL;
    if (memblock_alloc_device(ctx, &mem_6398, bytes_6395, "mem_6398")) {
        err = 1;
        goto cleanup;
    }
    
    struct memblock_device mem_6400;
    
    mem_6400.references = NULL;
    if (memblock_alloc_device(ctx, &mem_6400, bytes_6395, "mem_6400")) {
        err = 1;
        goto cleanup;
    }
    
    struct memblock_device mem_6402;
    
    mem_6402.references = NULL;
    if (memblock_alloc_device(ctx, &mem_6402, bytes_6395, "mem_6402")) {
        err = 1;
        goto cleanup;
    }
    
    int64_t bytes_6403 = (int64_t) 4 * implz2080U_6265;
    struct memblock_device mem_6404;
    
    mem_6404.references = NULL;
    if (memblock_alloc_device(ctx, &mem_6404, bytes_6403, "mem_6404")) {
        err = 1;
        goto cleanup;
    }
    
    struct memblock_device defunc_2_radix_sort_res_mem_6414;
    
    defunc_2_radix_sort_res_mem_6414.references = NULL;
    
    struct memblock_device mem_param_6391;
    
    mem_param_6391.references = NULL;
    if (memblock_set_device(ctx, &mem_param_6391, &n_mem_6386, "n_mem_6386") !=
        0)
        return 1;
    for (int32_t i_6278 = 0; i_6278 < iters_6268; i_6278++) {
        int32_t defunc_2_radix_sort_step_arg_6280 = mul32(2, i_6278);
        int32_t defunc_0_get_bit_arg_6281 = add32(1,
                                                  defunc_2_radix_sort_step_arg_6280);
        bool cond_6282 = defunc_0_get_bit_arg_6281 == 31;
        bool cond_6283 = defunc_2_radix_sort_step_arg_6280 == 31;
        
        if (slt64((int64_t) 0, implz2080U_6265)) {
            if (ctx->debugging)
                fprintf(ctx->log, "%s\n", "\n# SegScan");
            
            int64_t stage1_max_num_groups_6430;
            
            stage1_max_num_groups_6430 = ctx->opencl.max_group_size;
            
            int64_t stage1_num_groups_6431;
            
            stage1_num_groups_6431 = smin64(stage1_max_num_groups_6430,
                                            num_groups_6373);
            
            int32_t num_threads_6432;
            
            num_threads_6432 = sext_i64_i32(stage1_num_groups_6431 *
                segscan_group_sizze_6371);
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage1_6376,
                                                    1, smax64((int64_t) 1,
                                                              (int64_t) sizeof(int64_t) *
                                                              segscan_group_sizze_6371),
                                                    NULL));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage1_6376,
                                                    2, smax64((int64_t) 1,
                                                              (int64_t) sizeof(int64_t) *
                                                              segscan_group_sizze_6371),
                                                    NULL));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage1_6376,
                                                    3, smax64((int64_t) 1,
                                                              (int64_t) sizeof(int64_t) *
                                                              segscan_group_sizze_6371),
                                                    NULL));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage1_6376,
                                                    4, smax64((int64_t) 1,
                                                              (int64_t) sizeof(int64_t) *
                                                              segscan_group_sizze_6371),
                                                    NULL));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage1_6376,
                                                    5, sizeof(implz2080U_6265),
                                                    &implz2080U_6265));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage1_6376,
                                                    6,
                                                    sizeof(defunc_2_radix_sort_step_arg_6280),
                                                    &defunc_2_radix_sort_step_arg_6280));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage1_6376,
                                                    7,
                                                    sizeof(defunc_0_get_bit_arg_6281),
                                                    &defunc_0_get_bit_arg_6281));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage1_6376,
                                                    8, sizeof(cond_6282),
                                                    &cond_6282));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage1_6376,
                                                    9, sizeof(cond_6283),
                                                    &cond_6283));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage1_6376,
                                                    10,
                                                    sizeof(num_threads_6432),
                                                    &num_threads_6432));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage1_6376,
                                                    11,
                                                    sizeof(mem_param_6391.mem),
                                                    &mem_param_6391.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage1_6376,
                                                    12, sizeof(mem_6396.mem),
                                                    &mem_6396.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage1_6376,
                                                    13, sizeof(mem_6398.mem),
                                                    &mem_6398.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage1_6376,
                                                    14, sizeof(mem_6400.mem),
                                                    &mem_6400.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage1_6376,
                                                    15, sizeof(mem_6402.mem),
                                                    &mem_6402.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage1_6376,
                                                    16, sizeof(mem_6404.mem),
                                                    &mem_6404.mem));
            if (1 * ((size_t) stage1_num_groups_6431 *
                     (size_t) segscan_group_sizze_6371) != 0) {
                const size_t global_work_sizze_6525[1] =
                             {(size_t) stage1_num_groups_6431 *
                             (size_t) segscan_group_sizze_6371};
                const size_t local_work_sizze_6529[1] =
                             {segscan_group_sizze_6371};
                int64_t time_start_6526 = 0, time_end_6527 = 0;
                
                if (ctx->debugging) {
                    fprintf(ctx->log, "Launching %s with global work size [",
                            "main.scan_stage1_6376");
                    fprintf(ctx->log, "%zu", global_work_sizze_6525[0]);
                    fprintf(ctx->log, "] and local work size [");
                    fprintf(ctx->log, "%zu", local_work_sizze_6529[0]);
                    fprintf(ctx->log,
                            "]; local memory parameters sum to %d bytes.\n",
                            (int) (0 + smax64((int64_t) 1,
                                              (int64_t) sizeof(int64_t) *
                                              segscan_group_sizze_6371) +
                                   smax64((int64_t) 1,
                                          (int64_t) sizeof(int64_t) *
                                          segscan_group_sizze_6371) +
                                   smax64((int64_t) 1,
                                          (int64_t) sizeof(int64_t) *
                                          segscan_group_sizze_6371) +
                                   smax64((int64_t) 1,
                                          (int64_t) sizeof(int64_t) *
                                          segscan_group_sizze_6371)));
                    time_start_6526 = get_wall_time();
                }
                OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                ctx->mainziscan_stage1_6376,
                                                                1, NULL,
                                                                global_work_sizze_6525,
                                                                local_work_sizze_6529,
                                                                0, NULL,
                                                                ctx->profiling_paused ||
                                                                !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                          &ctx->mainziscan_stage1_6376_runs,
                                                                                                          &ctx->mainziscan_stage1_6376_total_runtime)));
                if (ctx->debugging) {
                    OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                    time_end_6527 = get_wall_time();
                    
                    long time_diff_6528 = time_end_6527 - time_start_6526;
                    
                    fprintf(ctx->log, "kernel %s runtime: %ldus\n",
                            "main.scan_stage1_6376", time_diff_6528);
                }
            }
            if (ctx->debugging)
                fprintf(ctx->log, "%s: %llu%c", "elems_per_group",
                        (long long) (segscan_group_sizze_6371 *
                                     sdiv_up64(implz2080U_6265,
                                               sext_i32_i64(num_threads_6432))),
                        '\n');
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage2_6376,
                                                    1, smax64((int64_t) 1,
                                                              (int64_t) sizeof(int64_t) *
                                                              stage1_num_groups_6431),
                                                    NULL));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage2_6376,
                                                    2, smax64((int64_t) 1,
                                                              (int64_t) sizeof(int64_t) *
                                                              stage1_num_groups_6431),
                                                    NULL));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage2_6376,
                                                    3, smax64((int64_t) 1,
                                                              (int64_t) sizeof(int64_t) *
                                                              stage1_num_groups_6431),
                                                    NULL));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage2_6376,
                                                    4, smax64((int64_t) 1,
                                                              (int64_t) sizeof(int64_t) *
                                                              stage1_num_groups_6431),
                                                    NULL));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage2_6376,
                                                    5, sizeof(implz2080U_6265),
                                                    &implz2080U_6265));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage2_6376,
                                                    6,
                                                    sizeof(stage1_num_groups_6431),
                                                    &stage1_num_groups_6431));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage2_6376,
                                                    7, sizeof(num_threads_6432),
                                                    &num_threads_6432));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage2_6376,
                                                    8, sizeof(mem_6396.mem),
                                                    &mem_6396.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage2_6376,
                                                    9, sizeof(mem_6398.mem),
                                                    &mem_6398.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage2_6376,
                                                    10, sizeof(mem_6400.mem),
                                                    &mem_6400.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage2_6376,
                                                    11, sizeof(mem_6402.mem),
                                                    &mem_6402.mem));
            if (1 * ((size_t) (int64_t) 1 * (size_t) stage1_num_groups_6431) !=
                0) {
                const size_t global_work_sizze_6530[1] = {(size_t) (int64_t) 1 *
                             (size_t) stage1_num_groups_6431};
                const size_t local_work_sizze_6534[1] =
                             {stage1_num_groups_6431};
                int64_t time_start_6531 = 0, time_end_6532 = 0;
                
                if (ctx->debugging) {
                    fprintf(ctx->log, "Launching %s with global work size [",
                            "main.scan_stage2_6376");
                    fprintf(ctx->log, "%zu", global_work_sizze_6530[0]);
                    fprintf(ctx->log, "] and local work size [");
                    fprintf(ctx->log, "%zu", local_work_sizze_6534[0]);
                    fprintf(ctx->log,
                            "]; local memory parameters sum to %d bytes.\n",
                            (int) (0 + smax64((int64_t) 1,
                                              (int64_t) sizeof(int64_t) *
                                              stage1_num_groups_6431) +
                                   smax64((int64_t) 1,
                                          (int64_t) sizeof(int64_t) *
                                          stage1_num_groups_6431) +
                                   smax64((int64_t) 1,
                                          (int64_t) sizeof(int64_t) *
                                          stage1_num_groups_6431) +
                                   smax64((int64_t) 1,
                                          (int64_t) sizeof(int64_t) *
                                          stage1_num_groups_6431)));
                    time_start_6531 = get_wall_time();
                }
                OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                ctx->mainziscan_stage2_6376,
                                                                1, NULL,
                                                                global_work_sizze_6530,
                                                                local_work_sizze_6534,
                                                                0, NULL,
                                                                ctx->profiling_paused ||
                                                                !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                          &ctx->mainziscan_stage2_6376_runs,
                                                                                                          &ctx->mainziscan_stage2_6376_total_runtime)));
                if (ctx->debugging) {
                    OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                    time_end_6532 = get_wall_time();
                    
                    long time_diff_6533 = time_end_6532 - time_start_6531;
                    
                    fprintf(ctx->log, "kernel %s runtime: %ldus\n",
                            "main.scan_stage2_6376", time_diff_6533);
                }
            }
            
            int32_t required_groups_6507;
            
            required_groups_6507 = sext_i64_i32(sdiv_up64(implz2080U_6265,
                                                          segscan_group_sizze_6371));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage3_6376,
                                                    1, sizeof(implz2080U_6265),
                                                    &implz2080U_6265));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage3_6376,
                                                    2, sizeof(num_groups_6373),
                                                    &num_groups_6373));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage3_6376,
                                                    3, sizeof(num_threads_6432),
                                                    &num_threads_6432));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage3_6376,
                                                    4,
                                                    sizeof(required_groups_6507),
                                                    &required_groups_6507));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage3_6376,
                                                    5, sizeof(mem_6396.mem),
                                                    &mem_6396.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage3_6376,
                                                    6, sizeof(mem_6398.mem),
                                                    &mem_6398.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage3_6376,
                                                    7, sizeof(mem_6400.mem),
                                                    &mem_6400.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainziscan_stage3_6376,
                                                    8, sizeof(mem_6402.mem),
                                                    &mem_6402.mem));
            if (1 * ((size_t) num_groups_6373 *
                     (size_t) segscan_group_sizze_6371) != 0) {
                const size_t global_work_sizze_6535[1] =
                             {(size_t) num_groups_6373 *
                             (size_t) segscan_group_sizze_6371};
                const size_t local_work_sizze_6539[1] =
                             {segscan_group_sizze_6371};
                int64_t time_start_6536 = 0, time_end_6537 = 0;
                
                if (ctx->debugging) {
                    fprintf(ctx->log, "Launching %s with global work size [",
                            "main.scan_stage3_6376");
                    fprintf(ctx->log, "%zu", global_work_sizze_6535[0]);
                    fprintf(ctx->log, "] and local work size [");
                    fprintf(ctx->log, "%zu", local_work_sizze_6539[0]);
                    fprintf(ctx->log,
                            "]; local memory parameters sum to %d bytes.\n",
                            (int) 0);
                    time_start_6536 = get_wall_time();
                }
                OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                ctx->mainziscan_stage3_6376,
                                                                1, NULL,
                                                                global_work_sizze_6535,
                                                                local_work_sizze_6539,
                                                                0, NULL,
                                                                ctx->profiling_paused ||
                                                                !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                          &ctx->mainziscan_stage3_6376_runs,
                                                                                                          &ctx->mainziscan_stage3_6376_total_runtime)));
                if (ctx->debugging) {
                    OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                    time_end_6537 = get_wall_time();
                    
                    long time_diff_6538 = time_end_6537 - time_start_6536;
                    
                    fprintf(ctx->log, "kernel %s runtime: %ldus\n",
                            "main.scan_stage3_6376", time_diff_6538);
                }
            }
        }
        
        int64_t read_res_6540;
        
        OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                     mem_6396.mem,
                                                     ctx->failure_is_an_option ? CL_FALSE : CL_TRUE,
                                                     i_6270 * sizeof(int64_t),
                                                     sizeof(int64_t),
                                                     &read_res_6540, 0, NULL,
                                                     ctx->profiling_paused ||
                                                     !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                               &ctx->copy_scalar_from_dev_runs,
                                                                                               &ctx->copy_scalar_from_dev_total_runtime)));
        if (ctx->failure_is_an_option && futhark_context_sync(ctx) != 0)
            return 1;
        
        int64_t last_res_6342 = read_res_6540;
        int64_t read_res_6541;
        
        OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                     mem_6398.mem,
                                                     ctx->failure_is_an_option ? CL_FALSE : CL_TRUE,
                                                     i_6270 * sizeof(int64_t),
                                                     sizeof(int64_t),
                                                     &read_res_6541, 0, NULL,
                                                     ctx->profiling_paused ||
                                                     !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                               &ctx->copy_scalar_from_dev_runs,
                                                                                               &ctx->copy_scalar_from_dev_total_runtime)));
        if (ctx->failure_is_an_option && futhark_context_sync(ctx) != 0)
            return 1;
        
        int64_t last_res_6343 = read_res_6541;
        int64_t read_res_6542;
        
        OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                     mem_6400.mem,
                                                     ctx->failure_is_an_option ? CL_FALSE : CL_TRUE,
                                                     i_6270 * sizeof(int64_t),
                                                     sizeof(int64_t),
                                                     &read_res_6542, 0, NULL,
                                                     ctx->profiling_paused ||
                                                     !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                               &ctx->copy_scalar_from_dev_runs,
                                                                                               &ctx->copy_scalar_from_dev_total_runtime)));
        if (ctx->failure_is_an_option && futhark_context_sync(ctx) != 0)
            return 1;
        
        int64_t last_res_6344 = read_res_6542;
        struct memblock_device mem_6406;
        
        mem_6406.references = NULL;
        if (memblock_alloc_device(ctx, &mem_6406, bytes_6403, "mem_6406")) {
            err = 1;
            goto cleanup;
        }
        if (implz2080U_6265 * (int64_t) sizeof(int32_t) > 0) {
            OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                         mem_param_6391.mem,
                                                         mem_6406.mem,
                                                         (int64_t) 0,
                                                         (int64_t) 0,
                                                         implz2080U_6265 *
                                                         (int64_t) sizeof(int32_t),
                                                         0, NULL,
                                                         ctx->profiling_paused ||
                                                         !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                   &ctx->copy_dev_to_dev_runs,
                                                                                                   &ctx->copy_dev_to_dev_total_runtime)));
            if (ctx->debugging)
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
        }
        if (ctx->debugging)
            fprintf(ctx->log, "%s\n", "\n# SegMap");
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainzisegmap_6378, 1,
                                                sizeof(implz2080U_6265),
                                                &implz2080U_6265));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainzisegmap_6378, 2,
                                                sizeof(last_res_6342),
                                                &last_res_6342));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainzisegmap_6378, 3,
                                                sizeof(last_res_6343),
                                                &last_res_6343));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainzisegmap_6378, 4,
                                                sizeof(last_res_6344),
                                                &last_res_6344));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainzisegmap_6378, 5,
                                                sizeof(mem_param_6391.mem),
                                                &mem_param_6391.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainzisegmap_6378, 6,
                                                sizeof(mem_6396.mem),
                                                &mem_6396.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainzisegmap_6378, 7,
                                                sizeof(mem_6398.mem),
                                                &mem_6398.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainzisegmap_6378, 8,
                                                sizeof(mem_6400.mem),
                                                &mem_6400.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainzisegmap_6378, 9,
                                                sizeof(mem_6402.mem),
                                                &mem_6402.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainzisegmap_6378, 10,
                                                sizeof(mem_6404.mem),
                                                &mem_6404.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->mainzisegmap_6378, 11,
                                                sizeof(mem_6406.mem),
                                                &mem_6406.mem));
        if (1 * ((size_t) segmap_usable_groups_6382 *
                 (size_t) segmap_group_sizze_6381) != 0) {
            const size_t global_work_sizze_6543[1] =
                         {(size_t) segmap_usable_groups_6382 *
                         (size_t) segmap_group_sizze_6381};
            const size_t local_work_sizze_6547[1] = {segmap_group_sizze_6381};
            int64_t time_start_6544 = 0, time_end_6545 = 0;
            
            if (ctx->debugging) {
                fprintf(ctx->log, "Launching %s with global work size [",
                        "main.segmap_6378");
                fprintf(ctx->log, "%zu", global_work_sizze_6543[0]);
                fprintf(ctx->log, "] and local work size [");
                fprintf(ctx->log, "%zu", local_work_sizze_6547[0]);
                fprintf(ctx->log,
                        "]; local memory parameters sum to %d bytes.\n",
                        (int) 0);
                time_start_6544 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->mainzisegmap_6378,
                                                            1, NULL,
                                                            global_work_sizze_6543,
                                                            local_work_sizze_6547,
                                                            0, NULL,
                                                            ctx->profiling_paused ||
                                                            !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                      &ctx->mainzisegmap_6378_runs,
                                                                                                      &ctx->mainzisegmap_6378_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_6545 = get_wall_time();
                
                long time_diff_6546 = time_end_6545 - time_start_6544;
                
                fprintf(ctx->log, "kernel %s runtime: %ldus\n",
                        "main.segmap_6378", time_diff_6546);
            }
        }
        
        struct memblock_device mem_param_tmp_6428;
        
        mem_param_tmp_6428.references = NULL;
        if (memblock_set_device(ctx, &mem_param_tmp_6428, &mem_6406,
                                "mem_6406") != 0)
            return 1;
        if (memblock_set_device(ctx, &mem_param_6391, &mem_param_tmp_6428,
                                "mem_param_tmp_6428") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_param_tmp_6428,
                                  "mem_param_tmp_6428") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_6406, "mem_6406") != 0)
            return 1;
    }
    if (memblock_set_device(ctx, &defunc_2_radix_sort_res_mem_6414,
                            &mem_param_6391, "mem_param_6391") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_6396, "mem_6396") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_6398, "mem_6398") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_6400, "mem_6400") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_6402, "mem_6402") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_6404, "mem_6404") != 0)
        return 1;
    
    struct memblock_device mem_6416;
    
    mem_6416.references = NULL;
    if (memblock_alloc_device(ctx, &mem_6416, bytes_6403, "mem_6416")) {
        err = 1;
        goto cleanup;
    }
    if (implz2080U_6265 * (int64_t) sizeof(int32_t) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                     defunc_2_radix_sort_res_mem_6414.mem,
                                                     mem_6416.mem, (int64_t) 0,
                                                     (int64_t) 0,
                                                     implz2080U_6265 *
                                                     (int64_t) sizeof(int32_t),
                                                     0, NULL,
                                                     ctx->profiling_paused ||
                                                     !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                               &ctx->copy_dev_to_dev_runs,
                                                                                               &ctx->copy_dev_to_dev_total_runtime)));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    if (memblock_unref_device(ctx, &defunc_2_radix_sort_res_mem_6414,
                              "defunc_2_radix_sort_res_mem_6414") != 0)
        return 1;
    if (memblock_set_device(ctx, &out_mem_6426, &mem_6416, "mem_6416") != 0)
        return 1;
    (*out_mem_p_6524).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_6524, &out_mem_6426,
                            "out_mem_6426") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_6416, "mem_6416") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_param_6391, "mem_param_6391") != 0)
        return 1;
    if (memblock_unref_device(ctx, &defunc_2_radix_sort_res_mem_6414,
                              "defunc_2_radix_sort_res_mem_6414") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_6404, "mem_6404") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_6402, "mem_6402") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_6400, "mem_6400") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_6398, "mem_6398") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_6396, "mem_6396") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_6426, "out_mem_6426") != 0)
        return 1;
    
  cleanup:
    { }
    return err;
}
struct futhark_i32_1d {
    struct memblock_device mem;
    int64_t shape[1];
} ;
struct futhark_i32_1d *futhark_new_i32_1d(struct futhark_context *ctx, const
                                          int32_t *data, int64_t dim0)
{
    struct futhark_i32_1d *bad = NULL;
    struct futhark_i32_1d *arr =
                          (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, (size_t) dim0 * sizeof(int32_t),
                              "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    if ((size_t) dim0 * sizeof(int32_t) > 0)
        OPENCL_SUCCEED_OR_RETURN(clEnqueueWriteBuffer(ctx->opencl.queue,
                                                      arr->mem.mem, CL_TRUE, 0,
                                                      (size_t) dim0 *
                                                      sizeof(int32_t), data + 0,
                                                      0, NULL,
                                                      ctx->profiling_paused ||
                                                      !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                &ctx->copy_dev_to_host_runs,
                                                                                                &ctx->copy_dev_to_host_total_runtime)));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_i32_1d *futhark_new_raw_i32_1d(struct futhark_context *ctx, const
                                              cl_mem data, int offset,
                                              int64_t dim0)
{
    struct futhark_i32_1d *bad = NULL;
    struct futhark_i32_1d *arr =
                          (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, (size_t) dim0 * sizeof(int32_t),
                              "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    if ((size_t) dim0 * sizeof(int32_t) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue, data,
                                                     arr->mem.mem, offset, 0,
                                                     (size_t) dim0 *
                                                     sizeof(int32_t), 0, NULL,
                                                     ctx->profiling_paused ||
                                                     !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                               &ctx->copy_dev_to_dev_runs,
                                                                                               &ctx->copy_dev_to_dev_total_runtime)));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_i32_1d(struct futhark_context *ctx, struct futhark_i32_1d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref_device(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_i32_1d(struct futhark_context *ctx,
                          struct futhark_i32_1d *arr, int32_t *data)
{
    lock_lock(&ctx->lock);
    if ((size_t) arr->shape[0] * sizeof(int32_t) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                     arr->mem.mem,
                                                     ctx->failure_is_an_option ? CL_FALSE : CL_TRUE,
                                                     0, (size_t) arr->shape[0] *
                                                     sizeof(int32_t), data + 0,
                                                     0, NULL,
                                                     ctx->profiling_paused ||
                                                     !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                               &ctx->copy_host_to_dev_runs,
                                                                                               &ctx->copy_host_to_dev_total_runtime)));
        if (ctx->failure_is_an_option && futhark_context_sync(ctx) != 0)
            return 1;
    }
    lock_unlock(&ctx->lock);
    return 0;
}
cl_mem futhark_values_raw_i32_1d(struct futhark_context *ctx,
                                 struct futhark_i32_1d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
const int64_t *futhark_shape_i32_1d(struct futhark_context *ctx,
                                    struct futhark_i32_1d *arr)
{
    (void) ctx;
    return arr->shape;
}
int futhark_entry_main(struct futhark_context *ctx,
                       struct futhark_i32_1d **out0, const
                       struct futhark_i32_1d *in0)
{
    struct memblock_device n_mem_6386;
    
    n_mem_6386.references = NULL;
    
    int64_t implz2080U_6265;
    struct memblock_device out_mem_6426;
    
    out_mem_6426.references = NULL;
    
    int ret = 0;
    
    lock_lock(&ctx->lock);
    n_mem_6386 = in0->mem;
    implz2080U_6265 = in0->shape[0];
    if (!(implz2080U_6265 == in0->shape[0])) {
        ret = 1;
        if (!ctx->error)
            ctx->error =
                msgprintf("Error: entry point arguments have invalid sizes.\n");
    } else {
        ret = futrts_main(ctx, &out_mem_6426, n_mem_6386, implz2080U_6265);
        if (ret == 0) {
            assert((*out0 =
                    (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d))) !=
                NULL);
            (*out0)->mem = out_mem_6426;
            (*out0)->shape[0] = implz2080U_6265;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
