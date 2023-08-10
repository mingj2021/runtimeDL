add_custom_target(samples)

set(SAMPLES_LIST 
sampleAsyscTRTYolo  
)

foreach(f ${SAMPLES_LIST})
    add_subdirectory(${f})
endforeach(f SAMPLES_LIST)