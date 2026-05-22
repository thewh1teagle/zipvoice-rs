/* AUTO-GENERATED WITH CBINDGEN. DO NOT EDIT BY HAND.
   REGENERATE WITH:
   cbindgen crates/zipvoice-capi --crate zipvoice-capi --output crates/zipvoice-capi/include/zipvoice.h
*/

#ifndef ZIPVOICE_H
#define ZIPVOICE_H

#pragma once

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct ZipVoiceModel ZipVoiceModel;

const char *zipvoice_last_error(void);

ZipVoiceModel *zipvoice_model_load(const char *zipvoice_path, const char *vocos_path);

void zipvoice_model_free(ZipVoiceModel *model);

int zipvoice_generate_wav(ZipVoiceModel *model,
                          const char *ref_wav,
                          const char *ref_phonemes,
                          const char *target_phonemes,
                          float speed,
                          size_t num_steps,
                          float t_shift,
                          float guidance_scale,
                          uint64_t seed,
                          bool verbose,
                          const char *output_path);

#endif  /* ZIPVOICE_H */
