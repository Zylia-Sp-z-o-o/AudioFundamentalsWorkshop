ffmpeg -i sintel_trailer-audio.flac \
  -vn -ac 2 -ar 48000 -c:a aac -b:a 64k \
  -hls_time 4 \
  -hls_playlist_type vod \
  -hls_segment_filename audio_64k_%03d.aac \
  audio_64k.m3u8
