ffmpeg -i sintel_trailer-audio.flac \
  -vn -ac 2 -ar 48000 -c:a aac -b:a 128k \
  -hls_time 4 \
  -hls_playlist_type vod \
  -hls_segment_filename audio_128k_%03d.aac \
  audio_128k.m3u8
