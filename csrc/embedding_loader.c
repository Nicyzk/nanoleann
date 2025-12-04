#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/stat.h>
#include <string.h>

/**
 * Helper: Convert IEEE 754 half-precision (16-bit) to single-precision (32-bit).
 * This allows us to print and use the numbers in standard C.
 */
float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x0001;
    uint32_t exp  = (h >> 10) & 0x001f;
    uint32_t mant = h & 0x03ff;
    uint32_t f_bits;

    if (exp == 0) {
        if (mant == 0) {
            // Zero
            f_bits = (sign << 31);
        } else {
            // Denormalized number (simplification: treat as very small 0 for this demo, 
            // or perform complex shift to normalize)
            // For rigorous math, use full denormal handling. 
            // Here we flush denormals to zero for simplicity in a basic loader.
            f_bits = (sign << 31); 
        }
    } else if (exp == 31) {
        // Infinity or NaN
        f_bits = (sign << 31) | 0x7f800000 | (mant << 13);
    } else {
        // Normalized number
        // Re-bias exponent: Half bias is 15, Float bias is 127.
        // New exp = old_exp - 15 + 127 = old_exp + 112
        f_bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }

    // Reinterpret bits as float
    float result;
    memcpy(&result, &f_bits, sizeof(result));
    return result;
}

/**
 * Loads a binary embedding file containing FP16 data.
 * Returns a standard float* array (32-bit) for easy usage.
 */
float* load_embedding(const char* filename, size_t* out_count) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        perror("Failed to open file");
        return NULL;
    }

    // 1. Read the Item Size
    uint32_t item_size;
    if (fread(&item_size, sizeof(uint32_t), 1, f) != 1) {
        fprintf(stderr, "Error: Failed to read header.\n");
        fclose(f);
        return NULL;
    }

    // 2. Validate Type Compatibility
    // FP16 takes up 2 bytes.
    if (item_size != 2) {
        fprintf(stderr, "Error: Type mismatch!\n");
        fprintf(stderr, "  File contains %u bytes/element.\n", item_size);
        fprintf(stderr, "  Expected 2 bytes (FP16).\n");
        fclose(f);
        return NULL;
    }

    // 3. Determine number of elements
    struct stat sb;
    if (stat(filename, &sb) == -1) {
        perror("Failed to get file stats");
        fclose(f);
        return NULL;
    }

    long data_size_bytes = sb.st_size - sizeof(uint32_t);
    size_t count = data_size_bytes / item_size; // item_size is 2

    // 4. Allocate Memory for standard FLOAT (4 bytes)
    // We expand 16-bit storage to 32-bit memory for usage
    float* data = (float*)malloc(count * sizeof(float));
    if (!data) {
        perror("Memory allocation failed");
        fclose(f);
        return NULL;
    }

    // 5. Read and Convert
    // We read into a temporary buffer first or read one by one.
    // Reading in chunks is faster.
    uint16_t* temp_buf = (uint16_t*)malloc(data_size_bytes);
    if (!temp_buf) {
        perror("Temp memory failed");
        free(data);
        fclose(f);
        return NULL;
    }

    size_t read_count = fread(temp_buf, item_size, count, f);
    if (read_count != count) {
        fprintf(stderr, "Error: Read incomplete.\n");
        free(temp_buf);
        free(data);
        fclose(f);
        return NULL;
    }

    // Convert raw 16-bit bits to 32-bit floats
    for (size_t i = 0; i < count; i++) {
        data[i] = half_to_float(temp_buf[i]);
    }

    free(temp_buf); // Done with raw bits
    fclose(f);

    if (out_count) *out_count = count;
    return data;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <file.bin>\n", argv[0]);
        return 1;
    }

    size_t num_elements = 0;
    float* embedding = load_embedding(argv[1], &num_elements);

    if (embedding) {
        printf("Successfully loaded %lu values (converted FP16 -> Float32).\n", num_elements);
        
        printf("Preview: [");
        for (size_t i = 0; i < (num_elements < 5 ? num_elements : 5); i++) {
            // Now we can print with standard %f
            printf("%f, ", embedding[i]);
        }
        printf("...]\n");

        free(embedding);
    }

    return 0;
}