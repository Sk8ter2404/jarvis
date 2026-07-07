# JARVIS Action Index

> Machine-verified inventory of every dispatchable voice action — its handler, whether its
> result is spoken (INFORMATIVE = LLM restates / VERBATIM = spoken as-is / neither = only the
> preamble is heard), whether it has a `core/prompts.py` routing example, and whether a test
> references it. Regenerate with `python tools/gen_action_index.py`.

## Summary

| metric | count |
|---|---|
| Total registered actions (incl. aliases) | 487 |
| — monolith `ACTIONS` dict | 118 |
| — skill / core registered | 369 |
| VERBATIM speak set | 188 |
| INFORMATIVE speak set | 64 |
| neither set | 235 |
| no `prompts.py` example | 377 |
| no test reference | 70 |

A result in **neither** set is spoken only if the handler self-speaks; otherwise the answer
is dropped. That is correct for side-effect actions but is the recurring "logged but never
voiced" bug for read-outs — see the audit that seeded the 2026-07 read-out completeness sweep.

## Full index

Aliases sharing a handler are collapsed. `ex?` = has a prompts.py `[ACTION: …]` example.

| action(s) | handler | speak | ex? | tests |
|---|---|---|:--:|:--:|
| `show_last_diagnostic` | `core/actions.py:1011` | **VERBATIM** | — | 2 |
| `play_streaming` | `core/actions.py:1032` | neither | — | 1 |
| `click` | `core/actions.py:1058` | neither | yes | 3 |
| `youtube` | `core/actions.py:107` | neither | yes | 3 |
| `hotkey` | `core/actions.py:1096` | neither | — | 2 |
| `stop_pipeline` | `core/actions.py:1124` | neither | — | 1 |
| `force_backup` | `core/actions.py:1155` | neither | — | 2 |
| `get_time` | `core/actions.py:117` | *INFORMATIVE* | — | 4 |
| `reset_memory` | `core/actions.py:1178` | neither | — | 4 |
| `version_info`, `what_version`, `when_updated` | `core/actions.py:1206` | **VERBATIM** | — | 2 |
| `screenshot` | `core/actions.py:126` | *INFORMATIVE* | — | 5 |
| `check_for_updates`, `check_updates`, `is_there_an_update` | `core/actions.py:1260` | **VERBATIM** | yes | 1 |
| `file_a_bug`, `log_a_bug`, `report_a_bug`, `report_bug` | `core/actions.py:1270` | **VERBATIM** | yes | 2 |
| `run_smoke_test` | `core/actions.py:1303` | neither | — | 2 |
| `test_each_skill` | `core/actions.py:1346` | neither | — | 2 |
| `forget_last_hour` | `core/actions.py:1394` | neither | — | 2 |
| `latency_benchmark` | `core/actions.py:1428` | neither | — | 2 |
| `play_music` | `core/actions.py:1456` | *INFORMATIVE* | yes | 16 |
| `where_is_user` | `core/actions.py:1500` | *INFORMATIVE* | — | 0 |
| `see_screen` | `core/actions.py:1544` | *INFORMATIVE* | yes | 10 |
| `replay_last_action` | `core/actions.py:1606` | neither | — | 1 |
| `media_next` | `core/actions.py:163` | neither | — | 2 |
| `run_shell` | `core/actions.py:1650` | neither | — | 3 |
| `see_user` | `core/actions.py:1712` | *INFORMATIVE* | — | 1 |
| `media_prev` | `core/actions.py:172` | neither | — | 2 |
| `which_monitor` | `core/actions.py:1798` | *INFORMATIVE* | — | 2 |
| `media_playpause` | `core/actions.py:181` | neither | yes | 2 |
| `session_memory_recall` | `core/actions.py:1874` | **VERBATIM** | — | 0 |
| `volume_up` | `core/actions.py:190` | neither | — | 4 |
| `last_screen`, `previous_screen`, `recall_screen`, `screen_history` | `core/actions.py:1936` | *INFORMATIVE* | yes | 0 |
| `volume_down` | `core/actions.py:199` | neither | yes | 3 |
| `read_changelog`, `show_changelog`, `what_changed`, `whats_new` | `core/actions.py:2003` | **VERBATIM** | yes | 1 |
| `recent_changes` | `core/actions.py:2003` | neither | — | 0 |
| `start_overnight_upgrade` | `core/actions.py:2069` | neither | — | 0 |
| `volume_mute` | `core/actions.py:208` | neither | yes | 2 |
| `open_on_monitor` | `core/actions.py:2105` | neither | — | 1 |
| `move_window_to_monitor` | `core/actions.py:2185` | neither | yes | 1 |
| `netflix` | `core/actions.py:220` | neither | yes | 6 |
| `prime_video` | `core/actions.py:224` | neither | — | 2 |
| `create_skill` | `core/actions.py:2248` | neither | yes | 0 |
| `disney_plus` | `core/actions.py:228` | neither | yes | 2 |
| `hulu` | `core/actions.py:232` | neither | yes | 1 |
| `upgrade` | `core/actions.py:2329` | neither | — | 4 |
| `max` | `core/actions.py:236` | neither | yes | 3 |
| `spotify` | `core/actions.py:240` | neither | yes | 7 |
| `exit_jarvis`, `power_off_jarvis`, `quit_jarvis`, `shut_down`, `shutdown_jarvis`, `turn_off_jarvis` | `core/actions.py:2413` | neither | — | 3 |
| `youtube_play` | `core/actions.py:244` | neither | yes | 1 |
| `switch_llm` | `core/actions.py:2482` | neither | — | 3 |
| `hide_hud` | `core/actions.py:251` | neither | — | 2 |
| `find_on_screen` | `core/actions.py:2535` | *INFORMATIVE* | yes | 0 |
| `clear_llm_cache`, `reset_llm_cache` | `core/actions.py:2550` | neither | — | 2 |
| `ambient_listening`, `ambient_mode`, `chappie_mode`, `silent_learning` | `core/actions.py:2557` | neither | — | 3 |
| `show_hud` | `core/actions.py:300` | neither | — | 1 |
| `toggle_hud` | `core/actions.py:307` | neither | — | 0 |
| `test_mic` | `core/actions.py:327` | neither | — | 2 |
| `test_tts` | `core/actions.py:331` | neither | — | 1 |
| `test_vision` | `core/actions.py:335` | neither | — | 1 |
| `clear_tasks` | `core/actions.py:341` | neither | yes | 1 |
| `session_resume` | `core/actions.py:370` | neither | — | 0 |
| `restart` | `core/actions.py:381` | neither | — | 3 |
| `switch_llm_picker` | `core/actions.py:404` | neither | — | 1 |
| `compare_models`, `llm_costs`, `model_costs`, `model_prices` | `core/actions.py:414` | **VERBATIM** | yes | 1 |
| `show_llm_stats` | `core/actions.py:422` | **VERBATIM** | — | 2 |
| `press` | `core/actions.py:440` | neither | — | 0 |
| `scroll` | `core/actions.py:449` | neither | — | 2 |
| `list_skills` | `core/actions.py:464` | *INFORMATIVE* | — | 0 |
| `apple_music` | `core/actions.py:477` | neither | yes | 7 |
| `launch_app` | `core/actions.py:501` | neither | — | 7 |
| `pause_music` | `core/actions.py:555` | *INFORMATIVE* | yes | 2 |
| `resume_music` | `core/actions.py:568` | *INFORMATIVE* | yes | 2 |
| `now_playing` | `core/actions.py:578` | *INFORMATIVE* | — | 0 |
| `open_apple_music` | `core/actions.py:640` | neither | yes | 1 |
| `music_status` | `core/actions.py:653` | *INFORMATIVE* | yes | 0 |
| `queue_task` | `core/actions.py:678` | *INFORMATIVE* | yes | 2 |
| `open_url` | `core/actions.py:70` | *INFORMATIVE* | — | 12 |
| `list_windows` | `core/actions.py:700` | *INFORMATIVE* | — | 0 |
| `focus_window` | `core/actions.py:712` | neither | — | 2 |
| `minimize_window` | `core/actions.py:745` | neither | yes | 0 |
| `close_window` | `core/actions.py:763` | neither | yes | 2 |
| `type` | `core/actions.py:792` | neither | yes | 15 |
| `search` | `core/actions.py:80` | neither | — | 1 |
| `web_search` | `core/actions.py:80` | *INFORMATIVE* | — | 1 |
| `next_song` | `core/actions.py:814` | *INFORMATIVE* | yes | 2 |
| `previous_song` | `core/actions.py:826` | *INFORMATIVE* | yes | 1 |
| `show_tasks` | `core/actions.py:838` | *INFORMATIVE* | yes | 1 |
| `reload_skills` | `core/actions.py:936` | neither | — | 2 |
| `show_recent_facts` | `core/actions.py:959` | **VERBATIM** | — | 2 |
| `export_memory` | `core/actions.py:978` | neither | — | 1 |
| `run_diagnostic` | `core/actions.py:999` | **VERBATIM** | — | 5 |
| `pause_diagnostics` | `core/diagnostic_daemons.py:1496` | neither | — | 0 |
| `resume_diagnostics` | `core/diagnostic_daemons.py:1500` | neither | — | 0 |
| `diagnostic_daemon_status`, `diagnostic_status` | `core/diagnostic_daemons.py:1504` | **VERBATIM** | — | 2 |
| `ecobee_list_devices` | `?` | *INFORMATIVE* | — | 1 |
| `gate_status`, `stability_gate_status` | `?` | **VERBATIM** | — | 2 |
| `list_schedule`, `show_schedules` | `?` | **VERBATIM** | — | 0 |
| `nest_list_devices` | `?` | *INFORMATIVE* | — | 1 |
| `remove_schedule`, `run_schedule`, `schedule_cron`, `when_condition` | `?` | neither | — | 0 |
| `ring_list_devices` | `?` | *INFORMATIVE* | — | 1 |
| `list_promises` | `core/memory.py:539` | **VERBATIM** | — | 2 |
| `cancel_promise` | `core/memory.py:559` | **VERBATIM** | — | 2 |
| `control_device`, `control_smart_home`, `smart_home_control` | `core/smart_home_router.py:740` | **VERBATIM** | yes | 4 |
| `control_light`, `control_plug`, `kasa_control` | `core/smart_home_router.py:740` | neither | — | 0 |
| `smart_home_devices`, `smart_home_list` | `core/smart_home_router.py:820` | neither | — | 2 |
| `smart_home_router_status` | `core/smart_home_router.py:835` | **VERBATIM** | — | 3 |
| `refresh_smart_home_router` | `core/smart_home_router.py:870` | neither | — | 2 |
| `morning_tabs` | `skills/_example_skill.py:13` | neither | — | 0 |
| `vscode_command` | `skills/_example_skill.py:23` | neither | — | 0 |
| `amazon_orders`, `check_amazon_orders`, `check_orders` | `skills/amazon_order_tracker.py:602` | *INFORMATIVE* | yes | 1 |
| `recent_deliveries`, `recent_delivery` | `skills/amazon_order_tracker.py:629` | *INFORMATIVE* | — | 1 |
| `amazon_tracking_status` | `skills/amazon_order_tracker.py:649` | **VERBATIM** | — | 2 |
| `ambient_listen_start` | `skills/ambient_listen.py:1389` | neither | — | 5 |
| `ambient_listen_stop` | `skills/ambient_listen.py:1418` | neither | — | 3 |
| `ambient_audio_start` | `skills/ambient_listen.py:1439` | neither | — | 1 |
| `ambient_audio_stop` | `skills/ambient_listen.py:1465` | neither | — | 1 |
| `ambient_screen_start` | `skills/ambient_listen.py:1486` | neither | — | 1 |
| `ambient_screen_stop` | `skills/ambient_listen.py:1507` | neither | — | 1 |
| `ambient_full_start` | `skills/ambient_listen.py:1530` | neither | — | 1 |
| `ambient_full_stop` | `skills/ambient_listen.py:1539` | neither | — | 1 |
| `ambient_mic_only` | `skills/ambient_listen.py:1547` | neither | — | 1 |
| `ambient_listen_status` | `skills/ambient_listen.py:1559` | **VERBATIM** | — | 2 |
| `ambient_extract_start` | `skills/ambient_multimodal_extract.py:312` | neither | — | 1 |
| `ambient_extract_stop` | `skills/ambient_multimodal_extract.py:328` | neither | — | 1 |
| `ambient_extract_status` | `skills/ambient_multimodal_extract.py:349` | **VERBATIM** | — | 2 |
| `ambient_extract_now` | `skills/ambient_multimodal_extract.py:365` | neither | — | 1 |
| `anticipation_briefing_now` | `skills/anticipation_briefing.py:537` | neither | — | 1 |
| `anticipation_briefing_status` | `skills/anticipation_briefing.py:558` | **VERBATIM** | — | 2 |
| `anticipation_status` | `skills/anticipation_engine.py:551` | **VERBATIM** | — | 2 |
| `play_unheard` | `skills/apple_music_intel.py:638` | neither | yes | 1 |
| `play_vibe` | `skills/apple_music_intel.py:726` | neither | yes | 1 |
| `skip_track` | `skills/apple_music_intel.py:762` | neither | yes | 1 |
| `music_history` | `skills/apple_music_intel.py:823` | **VERBATIM** | yes | 2 |
| `music_taste` | `skills/apple_music_intel.py:839` | **VERBATIM** | — | 1 |
| `music_aggregate` | `skills/apple_music_intel.py:868` | **VERBATIM** | — | 2 |
| `audio_autoswitch_status` | `skills/audio_autoswitch.py:66` | **VERBATIM** | yes | 0 |
| `audio_autoswitch_on` | `skills/audio_autoswitch.py:77` | **VERBATIM** | yes | 0 |
| `audio_autoswitch_off` | `skills/audio_autoswitch.py:83` | **VERBATIM** | yes | 0 |
| `switch_to_headset`, `use_headset` | `skills/audio_autoswitch.py:88` | **VERBATIM** | yes | 0 |
| `switch_to_speakers`, `use_speakers` | `skills/audio_autoswitch.py:96` | **VERBATIM** | yes | 0 |
| `print_status` | `skills/bambu_h2d_voice_companion.py:424` | **VERBATIM** | — | 2 |
| `check_print` | `skills/bambu_monitor.py:960` | *INFORMATIVE* | yes | 2 |
| `how_is_the_print`, `print_details` | `skills/bambu_monitor.py:995` | *INFORMATIVE* | yes | 2 |
| `pause_print` | `skills/bambu_print_announcer.py:486` | neither | — | 1 |
| `resume_print` | `skills/bambu_print_announcer.py:498` | neither | — | 1 |
| `proactive_announcer_status` | `skills/bambu_print_announcer.py:510` | **VERBATIM** | — | 2 |
| `bambu_setup`, `configure_printer`, `first_time_printer_setup`, `setup_bambu`, `setup_printer` | `skills/bambu_setup.py:518` | neither | — | 1 |
| `banter_status` | `skills/banter.py:566` | **VERBATIM** | — | 2 |
| `camera_status` | `skills/camera_system.py:402` | **VERBATIM** | — | 1 |
| `situational_awareness`, `where_am_i` | `skills/camera_system.py:465` | **VERBATIM** | — | 1 |
| `look_around` | `skills/camera_system.py:655` | **VERBATIM** | — | 2 |
| `chappie_recall_entity` | `skills/chappie_consciousness.py:538` | **VERBATIM** | — | 1 |
| `chappie_recall_today` | `skills/chappie_consciousness.py:567` | **VERBATIM** | — | 2 |
| `chappie_status` | `skills/chappie_consciousness.py:609` | **VERBATIM** | — | 2 |
| `compute`, `eval_python`, `python`, `run_python` | `skills/code_executor.py:395` | neither | yes | 4 |
| `reset_kernel` | `skills/code_executor.py:408` | neither | — | 1 |
| `check_credits` | `skills/credits_monitor.py:215` | *INFORMATIVE* | yes | 3 |
| `set_tts_backend` | `skills/custom_voice.py:507` | neither | yes | 1 |
| `list_tts_backends` | `skills/custom_voice.py:511` | **VERBATIM** | — | 2 |
| `enroll_xtts_sample` | `skills/custom_voice.py:521` | neither | — | 1 |
| `daily_briefing` | `skills/daily_briefing.py:444` | neither | — | 1 |
| `daily_recap` | `skills/daily_recap.py:735` | neither | yes | 1 |
| `check_budget` | `skills/disk_budget_watchdog.py:167` | **VERBATIM** | yes | 2 |
| `focus_mode` | `skills/dnd_focus_mode.py:509` | neither | yes | 3 |
| `end_focus_mode` | `skills/dnd_focus_mode.py:516` | neither | yes | 1 |
| `focus_mode_status` | `skills/dnd_focus_mode.py:519` | **VERBATIM** | — | 2 |
| `dossier`, `dossier_on`, `file_on`, `pull_up_dossier`, `pull_up_file`, `what_do_you_have_on`, `whats_on_file` | `skills/dossier.py:579` | neither | yes | 1 |
| `draft_preview_gate_status`, `outbound_gate_status` | `skills/draft_preview_gate.py:227` | **VERBATIM** | — | 2 |
| `list_emails`, `list_unread`, `unread_email`, `unread_emails` | `skills/email_triage.py:1034` | *INFORMATIVE* | — | 1 |
| `read_email`, `read_message`, `read_thread` | `skills/email_triage.py:1057` | *INFORMATIVE* | — | 1 |
| `compose_reply`, `draft_reply`, `pre_draft_reply` | `skills/email_triage.py:1078` | *INFORMATIVE* | — | 1 |
| `confirm_pending_draft`, `send_draft`, `send_pending_draft` | `skills/email_triage.py:1127` | neither | — | 2 |
| `discard_draft`, `scrap_pending_draft` | `skills/email_triage.py:1146` | neither | — | 1 |
| `edit_pending_draft` | `skills/email_triage.py:1155` | **VERBATIM** | — | 1 |
| `list_pending_drafts`, `pending_drafts` | `skills/email_triage.py:1179` | **VERBATIM** | — | 2 |
| `archive_email`, `archive_message` | `skills/email_triage.py:1189` | neither | — | 1 |
| `categorise_inbox`, `categorize_inbox`, `triage_inbox` | `skills/email_triage.py:1198` | **VERBATIM** | — | 1 |
| `email_briefing`, `inbox_briefing` | `skills/email_triage.py:1228` | **VERBATIM** | — | 1 |
| `email_triage_status` | `skills/email_triage.py:1269` | **VERBATIM** | — | 2 |
| `enroll_voice`, `learn_my_voice` | `skills/enroll_voice.py:240` | neither | yes | 1 |
| `identify_speaker`, `who_is_talking`, `whos_talking` | `skills/enroll_voice.py:265` | **VERBATIM** | — | 3 |
| `enrolled_voices`, `list_enrolled_voices` | `skills/enroll_voice.py:287` | **VERBATIM** | — | 2 |
| `forget_voice` | `skills/enroll_voice.py:301` | neither | — | 1 |
| `set_active_speaker` | `skills/enroll_voice.py:313` | neither | — | 1 |
| `voice_id_status` | `skills/enroll_voice.py:326` | **VERBATIM** | — | 2 |
| `evening_briefing` | `skills/evening_briefing.py:793` | neither | — | 3 |
| `enroll_face`, `learn_my_face`, `remember_my_face` | `skills/face_id.py:197` | neither | — | 1 |
| `learn_guest`, `learn_their_face`, `remember_their_face`, `remember_this_person` | `skills/face_id.py:257` | neither | — | 1 |
| `do_you_recognize_me`, `recognize_face`, `who_am_i`, `whoami`, `whos_at_the_desk` | `skills/face_id.py:305` | **VERBATIM** | yes | 2 |
| `face_id_status` | `skills/face_id.py:357` | **VERBATIM** | — | 1 |
| `forget_face` | `skills/face_id.py:395` | neither | — | 1 |
| `list_enrolled_faces` | `skills/face_id.py:416` | **VERBATIM** | — | 2 |
| `gaze_status` | `skills/face_tracker.py:1282` | **VERBATIM** | — | 2 |
| `gaze_stats` | `skills/face_tracker.py:1310` | **VERBATIM** | — | 2 |
| `face_track_status` | `skills/face_tracker.py:1340` | **VERBATIM** | — | 2 |
| `calibrate_gaze` | `skills/face_tracker.py:1430` | **VERBATIM** | — | 0 |
| `gaze_calibration_status` | `skills/face_tracker.py:1476` | **VERBATIM** | — | 1 |
| `forget_gaze_calibration` | `skills/face_tracker.py:1489` | **VERBATIM** | — | 0 |
| `gaze_tracking_on` | `skills/face_tracker.py:1499` | **VERBATIM** | yes | 0 |
| `gaze_tracking_off` | `skills/face_tracker.py:1515` | **VERBATIM** | — | 0 |
| `guard_on` | `skills/guard_mode.py:548` | neither | — | 1 |
| `guard_off` | `skills/guard_mode.py:580` | neither | — | 1 |
| `guard_status` | `skills/guard_mode.py:597` | **VERBATIM** | — | 2 |
| `hardware_sensors` | `skills/hardware_sensors.py:20` | **VERBATIM** | yes | 2 |
| `generate_image` | `skills/image_gen.py:361` | neither | — | 1 |
| `make_picture` | `skills/image_gen.py:386` | neither | yes | 1 |
| `play_playlist` | `skills/itunes_library.py:136` | neither | yes | 1 |
| `list_playlists` | `skills/itunes_library.py:176` | **VERBATIM** | yes | 2 |
| `shuffle_library` | `skills/itunes_library.py:203` | neither | yes | 1 |
| `keep_music_open` | `skills/itunes_library.py:333` | neither | yes | 1 |
| `stop_keeping_music_open` | `skills/itunes_library.py:369` | neither | yes | 1 |
| `air_mouse_on` | `skills/kinect_air_mouse.py:2379` | neither | — | 1 |
| `air_mouse_off` | `skills/kinect_air_mouse.py:2399` | neither | — | 1 |
| `air_mouse_status` | `skills/kinect_air_mouse.py:2419` | **VERBATIM** | — | 2 |
| `calibrate_air_mouse` | `skills/kinect_air_mouse.py:2439` | **VERBATIM** | — | 1 |
| `gesture_status` | `skills/kinect_gestures.py:492` | **VERBATIM** | — | 1 |
| `gestures_on` | `skills/kinect_gestures.py:511` | neither | — | 0 |
| `gestures_off` | `skills/kinect_gestures.py:535` | neither | — | 0 |
| `calibrate_pointing`, `point_calibrate` | `skills/kinect_pointing.py:352` | **VERBATIM** | — | 1 |
| `list_point_targets`, `point_targets` | `skills/kinect_pointing.py:397` | **VERBATIM** | — | 2 |
| `forget_point_target` | `skills/kinect_pointing.py:420` | neither | — | 1 |
| `point_at`, `point_control` | `skills/kinect_pointing.py:433` | **VERBATIM** | — | 1 |
| `point_status` | `skills/kinect_pointing.py:488` | **VERBATIM** | — | 2 |
| `point_control_on` | `skills/kinect_pointing.py:515` | neither | — | 1 |
| `point_control_off` | `skills/kinect_pointing.py:533` | neither | — | 1 |
| `who_is_here` | `skills/kinect_vision.py:102` | **VERBATIM** | — | 1 |
| `scan_room` | `skills/kinect_vision.py:133` | **VERBATIM** | — | 1 |
| `kinect_look` | `skills/kinect_vision.py:137` | *INFORMATIVE* | — | 1 |
| `what_do_you_see_kinect` | `skills/kinect_vision.py:179` | *INFORMATIVE* | — | 1 |
| `kinect_status` | `skills/kinect_vision.py:53` | **VERBATIM** | — | 1 |
| `local_describe_screen` | `skills/local_vision.py:108` | *INFORMATIVE* | yes | 1 |
| `local_click_target_by_description` | `skills/local_vision.py:297` | neither | — | 1 |
| `mcp_status` | `skills/mcp_tools.py:149` | **VERBATIM** | — | 2 |
| `mcp_list_tools` | `skills/mcp_tools.py:166` | **VERBATIM** | — | 1 |
| `mcp_call` | `skills/mcp_tools.py:190` | **VERBATIM** | yes | 1 |
| `mcp_reload` | `skills/mcp_tools.py:223` | **VERBATIM** | — | 1 |
| `list_models` | `skills/model_picker.py:339` | **VERBATIM** | yes | 1 |
| `current_model` | `skills/model_picker.py:383` | **VERBATIM** | yes | 1 |
| `set_model` | `skills/model_picker.py:405` | neither | yes | 1 |
| `set_brain` | `skills/model_picker.py:498` | neither | yes | 1 |
| `arrival_briefing`, `morning_arrival` | `skills/morning_arrival.py:854` | neither | yes | 2 |
| `arrival_briefing_v2`, `morning_arrival_v2` | `skills/morning_arrival_v2.py:681` | neither | — | 1 |
| `morning_briefing` | `skills/morning_briefing.py:453` | **VERBATIM** | yes | 8 |
| `morning_chain_pick` | `skills/morning_chain.py:310` | neither | — | 1 |
| `morning_handoff` | `skills/morning_handoff.py:736` | neither | yes | 2 |
| `predictive_morning_setup`, `setup_workspace`, `workspace_setup` | `skills/morning_handoff.py:744` | **VERBATIM** | yes | 2 |
| `calendar_next`, `calendar_today`, `ms_graph_calendar` | `skills/ms_graph.py:794` | **VERBATIM** | yes | 4 |
| `list_wifi_clients`, `network_clients`, `who_is_on_the_wifi` | `skills/network_deco.py:681` | neither | — | 1 |
| `who_is_on_wifi` | `skills/network_deco.py:681` | *INFORMATIVE* | yes | 1 |
| `is_printer_online` | `skills/network_deco.py:697` | **VERBATIM** | yes | 2 |
| `printer_online` | `skills/network_deco.py:697` | neither | — | 1 |
| `device_online` | `skills/network_deco.py:717` | neither | — | 1 |
| `is_device_online` | `skills/network_deco.py:717` | **VERBATIM** | yes | 1 |
| `bandwidth_hogs`, `network_usage`, `whats_using_bandwidth` | `skills/network_deco.py:739` | neither | — | 1 |
| `disable_guest_network`, `kick_guest_network` | `skills/network_deco.py:794` | neither | yes | 1 |
| `enable_guest_network` | `skills/network_deco.py:798` | neither | — | 1 |
| `deco_topology` | `skills/network_deco.py:802` | **VERBATIM** | — | 1 |
| `network_topology` | `skills/network_deco.py:802` | neither | — | 1 |
| `deco_status` | `skills/network_deco.py:818` | **VERBATIM** | — | 2 |
| `deco_refresh`, `refresh_network` | `skills/network_deco.py:832` | neither | — | 1 |
| `news_briefing` | `skills/news_briefing.py:382` | neither | yes | 4 |
| `enable_night_owl`, `night_owl_mode`, `night_owl_on` | `skills/night_owl_mode.py:423` | neither | yes | 2 |
| `disable_night_owl`, `end_night_owl`, `night_owl_off` | `skills/night_owl_mode.py:426` | neither | yes | 1 |
| `good_morning` | `skills/night_owl_mode.py:429` | neither | — | 1 |
| `night_owl_status` | `skills/night_owl_mode.py:437` | **VERBATIM** | yes | 2 |
| `notification_triage_status`, `triage_status` | `skills/notification_triage.py:1330` | **VERBATIM** | — | 2 |
| `list_notification_rules` | `skills/notification_triage.py:1356` | **VERBATIM** | — | 2 |
| `add_notification_rule` | `skills/notification_triage.py:1365` | neither | — | 1 |
| `remove_notification_rule` | `skills/notification_triage.py:1387` | neither | — | 1 |
| `list_recent_notifications`, `recent_notifications_summary` | `skills/notification_triage.py:1401` | *INFORMATIVE* | yes | 1 |
| `pause_notification_triage` | `skills/notification_triage.py:1419` | neither | — | 1 |
| `resume_notification_triage` | `skills/notification_triage.py:1423` | neither | — | 1 |
| `obs_start_recording` | `skills/obs_control.py:104` | neither | — | 1 |
| `obs_stop_recording` | `skills/obs_control.py:118` | neither | — | 1 |
| `obs_pause_recording` | `skills/obs_control.py:130` | neither | — | 1 |
| `obs_switch_scene` | `skills/obs_control.py:158` | neither | yes | 1 |
| `obs_toggle_mute` | `skills/obs_control.py:195` | neither | yes | 1 |
| `pattern_predictions` | `skills/pattern_learning.py:1061` | **VERBATIM** | yes | 2 |
| `pattern_offer_now` | `skills/pattern_learning.py:1079` | **VERBATIM** | — | 2 |
| `pattern_aggregate` | `skills/pattern_learning.py:1084` | **VERBATIM** | — | 2 |
| `weekly_digest` | `skills/pattern_learning.py:1093` | **VERBATIM** | — | 2 |
| `pattern_stats` | `skills/pattern_learning.py:1107` | **VERBATIM** | — | 2 |
| `rag_search` | `skills/personal_rag.py:118` | **VERBATIM** | yes | 1 |
| `rag_search_quiet` | `skills/personal_rag.py:135` | neither | — | 1 |
| `search_my_files` | `skills/personal_rag.py:135` | **VERBATIM** | — | 2 |
| `rag_reindex` | `skills/personal_rag.py:175` | neither | — | 1 |
| `rag_status` | `skills/personal_rag.py:193` | **VERBATIM** | — | 2 |
| `rag_configure` | `skills/personal_rag.py:212` | neither | — | 1 |
| `rag_open_top` | `skills/personal_rag.py:247` | neither | yes | 1 |
| `notify_phone`, `push_to_phone`, `text_my_phone` | `skills/phone_bridge.py:872` | neither | yes | 1 |
| `phone_bridge_status`, `phone_status` | `skills/phone_bridge.py:901` | **VERBATIM** | — | 2 |
| `list_phone_backends` | `skills/phone_bridge.py:933` | **VERBATIM** | — | 2 |
| `pause_phone_bridge` | `skills/phone_bridge.py:955` | neither | — | 1 |
| `resume_phone_bridge` | `skills/phone_bridge.py:960` | neither | — | 1 |
| `print_companion_status` | `skills/proactive_print_companion.py:716` | **VERBATIM** | — | 2 |
| `print_companion_history` | `skills/proactive_print_companion.py:740` | neither | — | 1 |
| `robot_status` | `skills/repo_robot.py:195` | **VERBATIM** | — | 2 |
| `robot_blocker` | `skills/repo_robot.py:234` | **VERBATIM** | yes | 1 |
| `next_robot_step` | `skills/repo_robot.py:254` | **VERBATIM** | yes | 1 |
| `schedule_recurring` | `skills/schedule_manager.py:147` | neither | yes | 1 |
| `schedule_once` | `skills/schedule_manager.py:238` | neither | yes | 0 |
| `schedule_when` | `skills/schedule_manager.py:264` | neither | yes | 0 |
| `list_schedules` | `skills/schedule_manager.py:295` | **VERBATIM** | — | 0 |
| `cancel_schedule` | `skills/schedule_manager.py:310` | neither | — | 0 |
| `fire_schedule` | `skills/schedule_manager.py:325` | neither | — | 0 |
| `schedule_status` | `skills/schedule_manager.py:337` | **VERBATIM** | — | 1 |
| `screen_watch_status` | `skills/screen_watch.py:323` | **VERBATIM** | — | 2 |
| `are_you_ok`, `self_diagnostic`, `system_check` | `skills/self_diagnostic.py:2666` | **VERBATIM** | — | 2 |
| `what_is_broken`, `whats_broken` | `skills/self_diagnostic.py:2781` | **VERBATIM** | — | 2 |
| `diagnostic_history` | `skills/self_diagnostic.py:2816` | **VERBATIM** | — | 2 |
| `last_diagnostic_run` | `skills/self_diagnostic.py:2841` | **VERBATIM** | — | 2 |
| `ecobee_request_pin` | `skills/sh_ecobee.py:316` | **VERBATIM** | — | 1 |
| `ecobee_complete_setup` | `skills/sh_ecobee.py:331` | neither | — | 1 |
| `ecobee_authorize` | `skills/sh_ecobee.py:369` | **VERBATIM** | — | 1 |
| `govee_list`, `govee_list_devices` | `skills/sh_govee.py:380` | *INFORMATIVE* | — | 1 |
| `hue_retry_connect` | `skills/sh_hue.py:233` | **VERBATIM** | — | 1 |
| `hue_list`, `hue_list_devices` | `skills/sh_hue.py:404` | *INFORMATIVE* | yes | 1 |
| `hue_set_bridge_ip` | `skills/sh_hue.py:418` | neither | — | 1 |
| `kasa_list`, `kasa_list_devices`, `tplink_list` | `skills/sh_kasa.py:311` | *INFORMATIVE* | — | 1 |
| `lifx_list`, `lifx_list_devices` | `skills/sh_lifx.py:226` | *INFORMATIVE* | — | 1 |
| `nest_authorize` | `skills/sh_nest.py:363` | **VERBATIM** | — | 1 |
| `ring_authorize` | `skills/sh_ring.py:299` | **VERBATIM** | — | 1 |
| `smart_life_list`, `tuya_list`, `tuya_list_devices` | `skills/sh_tuya.py:149` | *INFORMATIVE* | — | 1 |
| `discover_smart_home`, `refresh_smart_home`, `smart_home_discover`, `smart_home_setup` | `skills/smart_home_discover.py:1112` | neither | — | 1 |
| `list_smart_home_devices`, `smart_home_catalog` | `skills/smart_home_discover.py:1381` | **VERBATIM** | — | 2 |
| `forget_alexa_login`, `smart_home_purge_cookie` | `skills/smart_home_discover.py:1397` | neither | — | 1 |
| `last_gate_result`, `last_stability_gate`, `last_stability_gate_result` | `skills/stability_gate_status.py:56` | **VERBATIM** | — | 2 |
| `audio_music_status` | `skills/standby_audio_detect.py:538` | **VERBATIM** | — | 2 |
| `status_panel`, `suit_diagnostics`, `system_status` | `skills/status_panel.py:511` | **VERBATIM** | yes | 2 |
| `suit_up`, `suit_up_sequence` | `skills/suit_up.py:361` | neither | yes | 1 |
| `check_system` | `skills/system_monitor.py:249` | **VERBATIM** | yes | 3 |
| `status_report`, `system_pulse` | `skills/system_pulse.py:666` | **VERBATIM** | yes | 5 |
| `check_teams` | `skills/teams_nudge.py:203` | **VERBATIM** | yes | 3 |
| `set_timer` | `skills/timer.py:364` | neither | yes | 9 |
| `list_timers` | `skills/timer.py:401` | **VERBATIM** | — | 1 |
| `cancel_timer` | `skills/timer.py:427` | neither | yes | 1 |
| `calibrate_tv_region`, `tv_calibrate` | `skills/tv_detect.py:303` | **VERBATIM** | — | 1 |
| `tv_detect_status`, `tv_status` | `skills/tv_detect.py:348` | **VERBATIM** | — | 2 |
| `tv_detect_on` | `skills/tv_detect.py:381` | **VERBATIM** | yes | 1 |
| `tv_detect_off` | `skills/tv_detect.py:399` | **VERBATIM** | — | 1 |
| `wake_listener_start` | `skills/wake_listener.py:379` | neither | — | 1 |
| `wake_listener_stop` | `skills/wake_listener.py:396` | neither | — | 1 |
| `wake_listener_status` | `skills/wake_listener.py:410` | **VERBATIM** | — | 2 |
| `wake_listener_configure` | `skills/wake_listener.py:436` | neither | — | 1 |
| `guest_mode_on` | `skills/wake_listener.py:504` | neither | — | 1 |
| `guest_mode_off` | `skills/wake_listener.py:512` | neither | — | 1 |
| `voice_gating_on` | `skills/wake_listener.py:519` | neither | — | 1 |
| `voice_gating_off` | `skills/wake_listener.py:526` | neither | — | 1 |
| `weather_briefing`, `weather_forecast` | `skills/weather_briefing.py:504` | **VERBATIM** | yes | 5 |
| `weekly_digest_now` | `skills/weekly_digest_briefing.py:475` | neither | — | 1 |
| `weekly_digest_status` | `skills/weekly_digest_briefing.py:493` | **VERBATIM** | — | 2 |
| `wellness_status` | `skills/wellness.py:331` | **VERBATIM** | yes | 1 |
| `workshop_status` | `skills/workshop_mode.py:287` | **VERBATIM** | — | 2 |
| `youtube_direct`, `youtube_search_direct`, `yt_direct` | `skills/youtube_search.py:179` | neither | — | 1 |
