# JARVIS Action Index

> Machine-verified inventory of every dispatchable voice action — its handler, whether its
> result is spoken (INFORMATIVE = LLM restates / VERBATIM = spoken as-is / neither = only the
> preamble is heard), whether it has a `core/prompts.py` routing example, and whether a test
> references it. Regenerate with `python tools/gen_action_index.py`.

## Summary

| metric | count |
|---|---|
| Total registered actions (incl. aliases) | 671 |
| — monolith `ACTIONS` dict | 139 |
| — skill / core registered | 532 |
| VERBATIM speak set | 313 |
| INFORMATIVE speak set | 79 |
| neither set | 279 |
| no `prompts.py` example | 540 |
| no test reference | 86 |

A result in **neither** set is spoken only if the handler self-speaks; otherwise the answer
is dropped. That is correct for side-effect actions but is the recurring "logged but never
voiced" bug for read-outs — see the audit that seeded the 2026-07 read-out completeness sweep.

## Full index

Aliases sharing a handler are collapsed. `ex?` = has a prompts.py `[ACTION: …]` example.

| action(s) | handler | speak | ex? | tests |
|---|---|---|:--:|:--:|
| `wake_word_mode_off`, `wake_word_mode_on` | `bobert_companion.py:16805` | neither | — | 1 |
| `wake_word_mode_status` | `bobert_companion.py:16822` | **VERBATIM** | — | 3 |
| `ambient_learning_mode`, `ambient_learning_mode_off`, `ambient_learning_mode_on`, `enter_ambient_learning`, `exit_ambient_learning` | `bobert_companion.py:17380` | neither | — | 2 |
| `wake_resume_answer_then_quiet`, `wake_resume_stay_talkative` | `bobert_companion.py:17416` | neither | — | 2 |
| `ambient_mode_off`, `ambient_mode_on`, `silent_learning_off`, `silent_learning_on`, `start_eavesdropping`, `stop_eavesdropping` | `core/actions.py:1013` | neither | — | 4 |
| `greet_new_people_off`, `greet_new_people_on`, `notice_new_people`, `stop_greeting_people` | `core/actions.py:1058` | neither | — | 0 |
| `youtube` | `core/actions.py:107` | neither | yes | 4 |
| `reload_skills` | `core/actions.py:1089` | neither | — | 2 |
| `show_recent_facts` | `core/actions.py:1112` | **VERBATIM** | — | 2 |
| `export_memory` | `core/actions.py:1131` | neither | — | 1 |
| `run_diagnostic` | `core/actions.py:1152` | **VERBATIM** | — | 5 |
| `show_last_diagnostic` | `core/actions.py:1164` | **VERBATIM** | — | 2 |
| `get_time` | `core/actions.py:117` | *INFORMATIVE* | — | 5 |
| `play_streaming` | `core/actions.py:1185` | neither | — | 1 |
| `click` | `core/actions.py:1211` | neither | yes | 5 |
| `hotkey` | `core/actions.py:1249` | neither | — | 2 |
| `screenshot` | `core/actions.py:126` | *INFORMATIVE* | — | 5 |
| `stop_pipeline` | `core/actions.py:1277` | neither | — | 1 |
| `force_backup` | `core/actions.py:1308` | neither | — | 2 |
| `reset_memory` | `core/actions.py:1331` | neither | — | 4 |
| `version_info`, `what_version`, `when_updated` | `core/actions.py:1378` | **VERBATIM** | — | 4 |
| `check_for_updates`, `check_updates`, `is_there_an_update` | `core/actions.py:1449` | **VERBATIM** | yes | 1 |
| `file_a_bug`, `log_a_bug`, `report_a_bug`, `report_bug` | `core/actions.py:1459` | **VERBATIM** | yes | 2 |
| `run_smoke_test` | `core/actions.py:1492` | neither | — | 2 |
| `test_each_skill` | `core/actions.py:1535` | neither | — | 2 |
| `forget_last_hour` | `core/actions.py:1583` | neither | — | 2 |
| `latency_benchmark` | `core/actions.py:1653` | neither | — | 2 |
| `play_music` | `core/actions.py:1684` | *INFORMATIVE* | yes | 16 |
| `where_is_user` | `core/actions.py:1728` | *INFORMATIVE* | — | 0 |
| `see_screen` | `core/actions.py:1772` | *INFORMATIVE* | yes | 10 |
| `replay_last_action` | `core/actions.py:1834` | neither | — | 1 |
| `media_next` | `core/actions.py:186` | neither | — | 2 |
| `run_shell` | `core/actions.py:1878` | neither | — | 3 |
| `see_user` | `core/actions.py:1940` | *INFORMATIVE* | — | 1 |
| `media_prev` | `core/actions.py:195` | neither | — | 2 |
| `which_monitor` | `core/actions.py:2026` | *INFORMATIVE* | — | 2 |
| `media_playpause` | `core/actions.py:204` | neither | yes | 2 |
| `session_memory_recall` | `core/actions.py:2110` | **VERBATIM** | — | 0 |
| `volume_up` | `core/actions.py:213` | neither | — | 5 |
| `last_screen`, `previous_screen`, `recall_screen`, `screen_history` | `core/actions.py:2172` | *INFORMATIVE* | yes | 0 |
| `volume_down` | `core/actions.py:222` | neither | yes | 3 |
| `read_changelog`, `show_changelog`, `what_changed`, `whats_new` | `core/actions.py:2239` | **VERBATIM** | yes | 1 |
| `recent_changes` | `core/actions.py:2239` | neither | — | 0 |
| `start_overnight_upgrade` | `core/actions.py:2305` | neither | — | 0 |
| `volume_mute` | `core/actions.py:231` | neither | yes | 3 |
| `open_on_monitor` | `core/actions.py:2341` | neither | — | 1 |
| `set_volume` | `core/actions.py:240` | neither | yes | 2 |
| `move_window_to_monitor` | `core/actions.py:2421` | neither | yes | 1 |
| `create_skill` | `core/actions.py:2484` | neither | yes | 0 |
| `upgrade` | `core/actions.py:2574` | neither | — | 4 |
| `exit_jarvis`, `power_off_jarvis`, `quit_jarvis`, `shut_down`, `shutdown_jarvis`, `turn_off_jarvis` | `core/actions.py:2777` | neither | — | 4 |
| `switch_llm` | `core/actions.py:2858` | neither | — | 4 |
| `netflix` | `core/actions.py:286` | neither | yes | 7 |
| `prime_video` | `core/actions.py:290` | neither | — | 2 |
| `disney_plus` | `core/actions.py:294` | neither | yes | 2 |
| `find_on_screen` | `core/actions.py:2969` | *INFORMATIVE* | yes | 0 |
| `hulu` | `core/actions.py:298` | neither | yes | 1 |
| `clear_llm_cache`, `reset_llm_cache` | `core/actions.py:2984` | neither | — | 2 |
| `ambient_listening`, `ambient_mode`, `chappie_mode`, `silent_learning` | `core/actions.py:2991` | neither | — | 3 |
| `max` | `core/actions.py:302` | neither | yes | 3 |
| `spotify` | `core/actions.py:306` | neither | yes | 7 |
| `youtube_play` | `core/actions.py:310` | neither | yes | 1 |
| `hide_hud` | `core/actions.py:317` | neither | — | 3 |
| `show_hud` | `core/actions.py:366` | neither | — | 2 |
| `toggle_hud` | `core/actions.py:373` | neither | — | 1 |
| `test_mic` | `core/actions.py:393` | neither | — | 2 |
| `test_tts` | `core/actions.py:397` | neither | — | 1 |
| `test_vision` | `core/actions.py:401` | neither | — | 1 |
| `clear_tasks` | `core/actions.py:407` | neither | yes | 1 |
| `session_resume` | `core/actions.py:436` | neither | — | 0 |
| `restart` | `core/actions.py:447` | neither | — | 5 |
| `switch_llm_picker` | `core/actions.py:521` | neither | — | 1 |
| `compare_models`, `llm_costs`, `model_costs`, `model_prices` | `core/actions.py:531` | **VERBATIM** | yes | 1 |
| `show_llm_stats` | `core/actions.py:576` | **VERBATIM** | — | 2 |
| `press` | `core/actions.py:593` | neither | — | 0 |
| `scroll` | `core/actions.py:602` | neither | — | 3 |
| `list_skills` | `core/actions.py:617` | *INFORMATIVE* | — | 0 |
| `apple_music` | `core/actions.py:630` | neither | yes | 8 |
| `launch_app` | `core/actions.py:654` | neither | — | 7 |
| `open_url` | `core/actions.py:70` | *INFORMATIVE* | — | 13 |
| `pause_music` | `core/actions.py:708` | *INFORMATIVE* | yes | 2 |
| `resume_music` | `core/actions.py:721` | *INFORMATIVE* | yes | 2 |
| `now_playing` | `core/actions.py:731` | *INFORMATIVE* | — | 1 |
| `open_apple_music` | `core/actions.py:793` | neither | yes | 1 |
| `search` | `core/actions.py:80` | neither | — | 1 |
| `web_search` | `core/actions.py:80` | *INFORMATIVE* | — | 1 |
| `music_status` | `core/actions.py:806` | *INFORMATIVE* | yes | 0 |
| `queue_task` | `core/actions.py:831` | *INFORMATIVE* | yes | 2 |
| `list_windows` | `core/actions.py:853` | *INFORMATIVE* | — | 0 |
| `focus_window` | `core/actions.py:865` | neither | — | 2 |
| `minimize_window` | `core/actions.py:898` | neither | yes | 0 |
| `close_window` | `core/actions.py:916` | neither | yes | 2 |
| `type` | `core/actions.py:945` | neither | yes | 17 |
| `next_song` | `core/actions.py:967` | *INFORMATIVE* | yes | 2 |
| `previous_song` | `core/actions.py:979` | *INFORMATIVE* | yes | 1 |
| `show_tasks` | `core/actions.py:991` | *INFORMATIVE* | yes | 1 |
| `pause_diagnostics` | `core/diagnostic_daemons.py:1508` | neither | — | 0 |
| `resume_diagnostics` | `core/diagnostic_daemons.py:1512` | neither | — | 0 |
| `diagnostic_daemon_status`, `diagnostic_status` | `core/diagnostic_daemons.py:1516` | **VERBATIM** | — | 2 |
| `gate_status`, `stability_gate_status` | `?` | **VERBATIM** | — | 2 |
| `list_promises` | `core/memory.py:556` | **VERBATIM** | — | 2 |
| `cancel_promise` | `core/memory.py:576` | **VERBATIM** | — | 2 |
| `smart_home_router_status` | `core/smart_home_router.py:1000` | **VERBATIM** | — | 3 |
| `refresh_smart_home_router` | `core/smart_home_router.py:1035` | neither | — | 2 |
| `control_device`, `control_smart_home`, `smart_home_control` | `core/smart_home_router.py:871` | **VERBATIM** | yes | 4 |
| `control_light`, `control_plug`, `kasa_control` | `core/smart_home_router.py:871` | neither | — | 0 |
| `smart_home_devices`, `smart_home_list` | `core/smart_home_router.py:985` | neither | — | 2 |
| `resume` | `core/wake_word.py:352` | **VERBATIM** | — | 3 |
| `morning_tabs` | `skills/_example_skill.py:13` | neither | — | 0 |
| `vscode_command` | `skills/_example_skill.py:23` | neither | — | 0 |
| `air_control_on` | `skills/air_control.py:385` | **VERBATIM** | yes | 1 |
| `air_control_off` | `skills/air_control.py:405` | **VERBATIM** | yes | 1 |
| `air_control_status` | `skills/air_control.py:414` | **VERBATIM** | yes | 1 |
| `amazon_orders`, `check_amazon_orders`, `check_orders` | `skills/amazon_order_tracker.py:602` | *INFORMATIVE* | yes | 1 |
| `recent_deliveries`, `recent_delivery` | `skills/amazon_order_tracker.py:629` | *INFORMATIVE* | — | 1 |
| `amazon_tracking_status` | `skills/amazon_order_tracker.py:649` | **VERBATIM** | — | 2 |
| `ambient_listen_start` | `skills/ambient_listen.py:1479` | neither | — | 5 |
| `ambient_listen_stop` | `skills/ambient_listen.py:1508` | neither | — | 3 |
| `ambient_audio_start` | `skills/ambient_listen.py:1529` | neither | — | 1 |
| `ambient_audio_stop` | `skills/ambient_listen.py:1555` | neither | — | 1 |
| `ambient_screen_start` | `skills/ambient_listen.py:1576` | neither | — | 1 |
| `ambient_screen_stop` | `skills/ambient_listen.py:1597` | neither | — | 1 |
| `ambient_full_start` | `skills/ambient_listen.py:1620` | neither | — | 1 |
| `ambient_full_stop` | `skills/ambient_listen.py:1629` | neither | — | 1 |
| `ambient_mic_only` | `skills/ambient_listen.py:1637` | neither | — | 1 |
| `ambient_listen_status` | `skills/ambient_listen.py:1649` | **VERBATIM** | — | 2 |
| `ambient_extract_start` | `skills/ambient_multimodal_extract.py:320` | neither | — | 1 |
| `ambient_extract_stop` | `skills/ambient_multimodal_extract.py:336` | neither | — | 1 |
| `ambient_extract_status` | `skills/ambient_multimodal_extract.py:357` | **VERBATIM** | — | 2 |
| `ambient_extract_now` | `skills/ambient_multimodal_extract.py:373` | neither | — | 1 |
| `anticipation_briefing_now` | `skills/anticipation_briefing.py:545` | neither | — | 1 |
| `anticipation_briefing_status` | `skills/anticipation_briefing.py:566` | **VERBATIM** | — | 2 |
| `anticipation_status` | `skills/anticipation_engine.py:568` | **VERBATIM** | — | 2 |
| `play_unheard` | `skills/apple_music_intel.py:646` | **VERBATIM** | yes | 2 |
| `play_vibe` | `skills/apple_music_intel.py:734` | **VERBATIM** | yes | 2 |
| `skip_track` | `skills/apple_music_intel.py:770` | **VERBATIM** | yes | 2 |
| `music_history` | `skills/apple_music_intel.py:832` | **VERBATIM** | yes | 2 |
| `music_taste` | `skills/apple_music_intel.py:848` | **VERBATIM** | — | 1 |
| `music_aggregate` | `skills/apple_music_intel.py:877` | **VERBATIM** | — | 2 |
| `audio_autoswitch_status` | `skills/audio_autoswitch.py:66` | **VERBATIM** | yes | 0 |
| `audio_autoswitch_on` | `skills/audio_autoswitch.py:77` | **VERBATIM** | yes | 0 |
| `audio_autoswitch_off` | `skills/audio_autoswitch.py:83` | **VERBATIM** | yes | 0 |
| `switch_to_headset`, `use_headset` | `skills/audio_autoswitch.py:88` | **VERBATIM** | yes | 0 |
| `switch_to_speakers`, `use_speakers` | `skills/audio_autoswitch.py:96` | **VERBATIM** | yes | 0 |
| `print_status` | `skills/bambu_h2d_voice_companion.py:492` | **VERBATIM** | — | 2 |
| `check_print` | `skills/bambu_monitor.py:1037` | *INFORMATIVE* | yes | 2 |
| `how_is_the_print`, `print_details` | `skills/bambu_monitor.py:1072` | *INFORMATIVE* | yes | 2 |
| `pause_print` | `skills/bambu_print_announcer.py:496` | **VERBATIM** | — | 2 |
| `resume_print` | `skills/bambu_print_announcer.py:508` | **VERBATIM** | — | 2 |
| `proactive_announcer_status` | `skills/bambu_print_announcer.py:520` | **VERBATIM** | — | 3 |
| `bambu_setup`, `configure_printer`, `first_time_printer_setup`, `setup_bambu`, `setup_printer` | `skills/bambu_setup.py:532` | neither | — | 1 |
| `banter_status` | `skills/banter.py:582` | **VERBATIM** | — | 2 |
| `browser_do`, `browser_run`, `browser_task` | `skills/browser_agent.py:630` | *INFORMATIVE* | yes | 2 |
| `book_appointment` | `skills/browser_agent.py:642` | *INFORMATIVE* | — | 1 |
| `fill_form` | `skills/browser_agent.py:656` | *INFORMATIVE* | — | 1 |
| `browse_for` | `skills/browser_agent.py:673` | *INFORMATIVE* | — | 1 |
| `find_cheapest` | `skills/browser_agent.py:689` | *INFORMATIVE* | — | 1 |
| `browser_open` | `skills/browser_agent.py:706` | neither | — | 1 |
| `browser_screenshot` | `skills/browser_agent.py:722` | neither | — | 1 |
| `browser_status` | `skills/browser_agent.py:767` | neither | — | 1 |
| `browser_stop` | `skills/browser_agent.py:796` | neither | — | 1 |
| `browser_reset_profile` | `skills/browser_agent.py:826` | neither | — | 2 |
| `camera_status` | `skills/camera_system.py:435` | **VERBATIM** | — | 2 |
| `situational_awareness`, `where_am_i` | `skills/camera_system.py:499` | **VERBATIM** | — | 2 |
| `look_around` | `skills/camera_system.py:689` | **VERBATIM** | — | 3 |
| `chappie_recall_entity` | `skills/chappie_consciousness.py:538` | **VERBATIM** | — | 1 |
| `chappie_recall_today` | `skills/chappie_consciousness.py:567` | **VERBATIM** | — | 2 |
| `chappie_status` | `skills/chappie_consciousness.py:609` | **VERBATIM** | — | 2 |
| `compute`, `eval_python`, `python`, `run_python` | `skills/code_executor.py:395` | *INFORMATIVE* | yes | 5 |
| `reset_kernel` | `skills/code_executor.py:408` | neither | — | 1 |
| `check_credits` | `skills/credits_monitor.py:215` | *INFORMATIVE* | yes | 3 |
| `set_tts_backend` | `skills/custom_voice.py:511` | neither | yes | 1 |
| `list_tts_backends` | `skills/custom_voice.py:515` | **VERBATIM** | — | 2 |
| `enroll_xtts_sample` | `skills/custom_voice.py:525` | neither | — | 1 |
| `daily_briefing` | `skills/daily_briefing.py:452` | neither | — | 1 |
| `daily_recap` | `skills/daily_recap.py:735` | neither | yes | 1 |
| `check_budget` | `skills/disk_budget_watchdog.py:167` | **VERBATIM** | yes | 2 |
| `focus_mode_status` | `skills/dnd_focus_mode.py:544` | **VERBATIM** | — | 3 |
| `dossier`, `dossier_on`, `file_on`, `pull_up_dossier`, `pull_up_file`, `what_do_you_have_on`, `whats_on_file` | `skills/dossier.py:666` | neither | yes | 1 |
| `draft_preview_gate_status`, `outbound_gate_status` | `skills/draft_preview_gate.py:227` | **VERBATIM** | — | 2 |
| `list_emails`, `list_unread`, `unread_email`, `unread_emails` | `skills/email_triage.py:1045` | *INFORMATIVE* | — | 1 |
| `read_email`, `read_message`, `read_thread` | `skills/email_triage.py:1068` | *INFORMATIVE* | — | 1 |
| `compose_reply`, `draft_reply`, `pre_draft_reply` | `skills/email_triage.py:1089` | *INFORMATIVE* | — | 1 |
| `confirm_pending_draft`, `send_draft`, `send_pending_draft` | `skills/email_triage.py:1138` | **VERBATIM** | — | 3 |
| `discard_draft`, `scrap_pending_draft` | `skills/email_triage.py:1157` | **VERBATIM** | — | 2 |
| `edit_pending_draft` | `skills/email_triage.py:1166` | **VERBATIM** | — | 1 |
| `list_pending_drafts`, `pending_drafts` | `skills/email_triage.py:1190` | **VERBATIM** | — | 2 |
| `archive_email`, `archive_message` | `skills/email_triage.py:1200` | **VERBATIM** | — | 2 |
| `categorise_inbox`, `categorize_inbox`, `triage_inbox` | `skills/email_triage.py:1209` | **VERBATIM** | — | 1 |
| `email_briefing`, `inbox_briefing` | `skills/email_triage.py:1250` | **VERBATIM** | — | 1 |
| `email_triage_status` | `skills/email_triage.py:1294` | **VERBATIM** | — | 2 |
| `enroll_voice`, `learn_my_voice` | `skills/enroll_voice.py:265` | **VERBATIM** | yes | 2 |
| `identify_speaker`, `who_is_talking`, `whos_talking` | `skills/enroll_voice.py:290` | **VERBATIM** | — | 3 |
| `enrolled_voices`, `list_enrolled_voices` | `skills/enroll_voice.py:312` | **VERBATIM** | — | 2 |
| `forget_voice` | `skills/enroll_voice.py:326` | **VERBATIM** | — | 2 |
| `set_active_speaker` | `skills/enroll_voice.py:338` | **VERBATIM** | — | 2 |
| `voice_id_status` | `skills/enroll_voice.py:351` | **VERBATIM** | — | 2 |
| `evening_briefing` | `skills/evening_briefing.py:801` | neither | — | 3 |
| `enroll_face`, `learn_my_face` | `skills/face_id.py:197` | **VERBATIM** | — | 2 |
| `remember_my_face` | `skills/face_id.py:197` | neither | — | 1 |
| `learn_guest`, `learn_their_face`, `remember_their_face` | `skills/face_id.py:257` | neither | — | 1 |
| `remember_this_person` | `skills/face_id.py:257` | **VERBATIM** | — | 2 |
| `do_you_recognize_me`, `recognize_face`, `who_am_i`, `whoami`, `whos_at_the_desk` | `skills/face_id.py:305` | **VERBATIM** | yes | 2 |
| `face_id_status` | `skills/face_id.py:357` | **VERBATIM** | — | 1 |
| `forget_face` | `skills/face_id.py:395` | **VERBATIM** | — | 2 |
| `list_enrolled_faces` | `skills/face_id.py:416` | **VERBATIM** | — | 2 |
| `gaze_status` | `skills/face_tracker.py:1305` | **VERBATIM** | — | 2 |
| `gaze_stats` | `skills/face_tracker.py:1333` | **VERBATIM** | — | 2 |
| `face_track_status` | `skills/face_tracker.py:1363` | **VERBATIM** | — | 2 |
| `calibrate_gaze` | `skills/face_tracker.py:1453` | **VERBATIM** | — | 0 |
| `gaze_calibration_status` | `skills/face_tracker.py:1499` | **VERBATIM** | — | 1 |
| `forget_gaze_calibration` | `skills/face_tracker.py:1512` | **VERBATIM** | — | 0 |
| `gaze_tracking_on` | `skills/face_tracker.py:1522` | **VERBATIM** | yes | 0 |
| `gaze_tracking_off` | `skills/face_tracker.py:1538` | **VERBATIM** | — | 0 |
| `do_not_disturb`, `focus_mode_on`, `quiet_mode` | `skills/focus_mode.py:294` | **VERBATIM** | yes | 1 |
| `focus_mode` | `skills/focus_mode.py:294` | neither | yes | 4 |
| `end_focus_mode` | `skills/focus_mode.py:322` | neither | yes | 2 |
| `focus_mode_off` | `skills/focus_mode.py:322` | **VERBATIM** | yes | 1 |
| `whats_missed` | `skills/focus_mode.py:349` | **VERBATIM** | yes | 1 |
| `gpu_status`, `gpu_usage`, `show_vram`, `vram_status`, `whats_loaded` | `skills/gpu_usage.py:218` | **VERBATIM** | yes | 1 |
| `guard_on` | `skills/guard_mode.py:615` | **VERBATIM** | — | 2 |
| `guard_off` | `skills/guard_mode.py:649` | **VERBATIM** | — | 2 |
| `guard_status` | `skills/guard_mode.py:666` | **VERBATIM** | — | 2 |
| `hardware_sensors` | `skills/hardware_sensors.py:20` | **VERBATIM** | yes | 2 |
| `bambu_camera_off`, `hide_bambu_camera`, `hide_printer_camera` | `skills/holographic_overlay/__init__.py:1000` | neither | — | 0 |
| `bambu_camera`, `bambu_camera_toggle`, `camera_hud`, `print_camera`, `printer_cam`, `printer_camera` | `skills/holographic_overlay/__init__.py:1007` | neither | — | 0 |
| `bambu_camera_status` | `skills/holographic_overlay/__init__.py:1013` | **VERBATIM** | — | 1 |
| `show_workshop_hud`, `workshop_hud_on` | `skills/holographic_overlay/__init__.py:1147` | neither | — | 1 |
| `hide_workshop_hud`, `workshop_hud_off` | `skills/holographic_overlay/__init__.py:1152` | neither | — | 1 |
| `workshop_hud`, `workshop_hud_toggle` | `skills/holographic_overlay/__init__.py:1157` | neither | — | 2 |
| `workshop_hud_status` | `skills/holographic_overlay/__init__.py:1163` | **VERBATIM** | — | 2 |
| `print_hud_on`, `show_workshop_print_monitor`, `workshop_print_hud_on`, `workshop_print_monitor_on` | `skills/holographic_overlay/__init__.py:1402` | neither | — | 1 |
| `hide_workshop_print_monitor`, `print_hud_off`, `workshop_print_hud_off`, `workshop_print_monitor_off` | `skills/holographic_overlay/__init__.py:1409` | neither | — | 1 |
| `print_hud`, `workshop_print_hud`, `workshop_print_monitor`, `workshop_print_monitor_toggle` | `skills/holographic_overlay/__init__.py:1416` | neither | — | 1 |
| `workshop_print_monitor_status` | `skills/holographic_overlay/__init__.py:1422` | **VERBATIM** | — | 2 |
| `arc_reactor_hud`, `holo_hud_v2_on`, `show_holo_hud_v2` | `skills/holographic_overlay/__init__.py:1526` | neither | — | 1 |
| `hide_holo_hud_v2`, `holo_hud_v2_off` | `skills/holographic_overlay/__init__.py:1531` | neither | — | 1 |
| `holo_hud_v2`, `holo_hud_v2_toggle`, `holographic_hud_v2` | `skills/holographic_overlay/__init__.py:1536` | neither | — | 1 |
| `holo_hud_v2_status` | `skills/holographic_overlay/__init__.py:1542` | **VERBATIM** | — | 2 |
| `arc_reactor_status_on`, `pulse_hud_on`, `show_status_hud`, `status_hud_on`, `status_ring_on` | `skills/holographic_overlay/__init__.py:1660` | neither | — | 1 |
| `arc_reactor_status_off`, `hide_status_hud`, `pulse_hud_off`, `status_hud_off`, `status_ring_off` | `skills/holographic_overlay/__init__.py:1665` | neither | — | 1 |
| `arc_reactor_status`, `arc_reactor_status_hud`, `arc_reactor_status_toggle`, `pulse_hud`, `status_hud`, `status_ring` | `skills/holographic_overlay/__init__.py:1670` | neither | — | 2 |
| `arc_reactor_status_status` | `skills/holographic_overlay/__init__.py:1676` | **VERBATIM** | — | 2 |
| `hud_v2_on`, `show_hud_v2`, `show_status_ring_v2`, `stark_status_ring_on`, `status_ring_v2_on` | `skills/holographic_overlay/__init__.py:1809` | neither | — | 1 |
| `hide_hud_v2`, `hide_status_ring_v2`, `hud_v2_off`, `stark_status_ring_off`, `status_ring_v2_off` | `skills/holographic_overlay/__init__.py:1814` | neither | — | 1 |
| `hud_v2`, `hud_v2_toggle`, `stark_status_ring`, `stark_status_ring_toggle`, `status_ring_v2` | `skills/holographic_overlay/__init__.py:1819` | neither | — | 1 |
| `stark_status_ring_status` | `skills/holographic_overlay/__init__.py:1825` | **VERBATIM** | — | 2 |
| `holographic_on`, `hud_on`, `show_holo`, `show_holographic_overlay` | `skills/holographic_overlay/__init__.py:322` | neither | — | 2 |
| `dismiss_holo`, `hide_holo`, `hide_holographic_overlay`, `holographic_off`, `hud_off` | `skills/holographic_overlay/__init__.py:327` | neither | — | 1 |
| `toggle_holo`, `toggle_holographic_overlay` | `skills/holographic_overlay/__init__.py:332` | neither | — | 1 |
| `arc_reactor` | `skills/holographic_overlay/__init__.py:464` | neither | — | 2 |
| `arc_reactor_on`, `holo_workshop`, `holo_workshop_canvas`, `workshop_canvas` | `skills/holographic_overlay/__init__.py:484` | neither | — | 1 |
| `arc_reactor_off` | `skills/holographic_overlay/__init__.py:489` | neither | — | 1 |
| `arc_reactor_pulse` | `skills/holographic_overlay/__init__.py:494` | neither | — | 1 |
| `bambu_overlay_on`, `show_bambu_overlay` | `skills/holographic_overlay/__init__.py:750` | neither | — | 1 |
| `bambu_overlay_off`, `hide_bambu_overlay` | `skills/holographic_overlay/__init__.py:757` | neither | — | 1 |
| `bambu_h2d_overlay`, `bambu_overlay`, `bambu_overlay_toggle` | `skills/holographic_overlay/__init__.py:764` | neither | — | 1 |
| `bambu_overlay_status` | `skills/holographic_overlay/__init__.py:770` | **VERBATIM** | — | 2 |
| `bambu_camera_on`, `show_bambu_camera`, `show_print_camera`, `show_printer_camera` | `skills/holographic_overlay/__init__.py:993` | neither | — | 0 |
| `generate_image` | `skills/image_gen.py:361` | **VERBATIM** | — | 2 |
| `make_picture` | `skills/image_gen.py:386` | **VERBATIM** | yes | 2 |
| `play_playlist` | `skills/itunes_library.py:136` | **VERBATIM** | yes | 2 |
| `list_playlists` | `skills/itunes_library.py:176` | **VERBATIM** | yes | 2 |
| `shuffle_library` | `skills/itunes_library.py:203` | **VERBATIM** | yes | 2 |
| `keep_music_open` | `skills/itunes_library.py:333` | **VERBATIM** | yes | 2 |
| `stop_keeping_music_open` | `skills/itunes_library.py:369` | **VERBATIM** | yes | 2 |
| `air_mouse_on` | `skills/kinect_air_mouse.py:3228` | **VERBATIM** | — | 2 |
| `air_mouse_off` | `skills/kinect_air_mouse.py:3248` | **VERBATIM** | — | 2 |
| `air_mouse_status` | `skills/kinect_air_mouse.py:3268` | **VERBATIM** | — | 2 |
| `air_mouse_arm`, `give_me_the_cursor`, `hand_mouse_on`, `mouse_control_on`, `take_the_cursor` | `skills/kinect_air_mouse.py:3289` | **VERBATIM** | yes | 3 |
| `air_mouse_disarm`, `hand_mouse_off`, `mouse_control_off`, `release_the_cursor` | `skills/kinect_air_mouse.py:3315` | **VERBATIM** | yes | 2 |
| `calibrate_air_mouse` | `skills/kinect_air_mouse.py:3338` | **VERBATIM** | — | 1 |
| `gesture_status` | `skills/kinect_gestures.py:501` | **VERBATIM** | — | 1 |
| `gestures_on` | `skills/kinect_gestures.py:520` | **VERBATIM** | — | 1 |
| `gestures_off` | `skills/kinect_gestures.py:544` | **VERBATIM** | — | 1 |
| `calibrate_pointing`, `point_calibrate` | `skills/kinect_pointing.py:352` | **VERBATIM** | — | 1 |
| `list_point_targets`, `point_targets` | `skills/kinect_pointing.py:397` | **VERBATIM** | — | 2 |
| `forget_point_target` | `skills/kinect_pointing.py:420` | **VERBATIM** | — | 2 |
| `point_at`, `point_control` | `skills/kinect_pointing.py:433` | **VERBATIM** | — | 1 |
| `point_status` | `skills/kinect_pointing.py:488` | **VERBATIM** | — | 2 |
| `point_control_on` | `skills/kinect_pointing.py:515` | **VERBATIM** | — | 2 |
| `point_control_off` | `skills/kinect_pointing.py:533` | **VERBATIM** | — | 2 |
| `who_is_here` | `skills/kinect_vision.py:103` | **VERBATIM** | — | 1 |
| `scan_room` | `skills/kinect_vision.py:134` | **VERBATIM** | — | 1 |
| `kinect_look` | `skills/kinect_vision.py:138` | *INFORMATIVE* | — | 1 |
| `what_do_you_see_kinect` | `skills/kinect_vision.py:180` | *INFORMATIVE* | — | 1 |
| `kinect_status` | `skills/kinect_vision.py:54` | **VERBATIM** | — | 1 |
| `local_describe_screen` | `skills/local_vision.py:121` | *INFORMATIVE* | yes | 1 |
| `local_click_target_by_description` | `skills/local_vision.py:310` | neither | — | 1 |
| `mcp_status` | `skills/mcp_tools.py:148` | **VERBATIM** | — | 2 |
| `mcp_list_tools` | `skills/mcp_tools.py:165` | **VERBATIM** | — | 1 |
| `mcp_call` | `skills/mcp_tools.py:189` | **VERBATIM** | yes | 1 |
| `mcp_reload` | `skills/mcp_tools.py:219` | **VERBATIM** | — | 1 |
| `list_models` | `skills/model_picker.py:411` | **VERBATIM** | yes | 1 |
| `current_model` | `skills/model_picker.py:455` | **VERBATIM** | yes | 1 |
| `set_model` | `skills/model_picker.py:477` | **VERBATIM** | yes | 2 |
| `set_brain` | `skills/model_picker.py:577` | **VERBATIM** | yes | 2 |
| `arrival_briefing`, `morning_arrival` | `skills/morning_arrival.py:854` | neither | yes | 2 |
| `arrival_briefing_v2`, `morning_arrival_v2` | `skills/morning_arrival_v2.py:686` | neither | — | 1 |
| `morning_briefing` | `skills/morning_briefing.py:448` | **VERBATIM** | yes | 8 |
| `morning_chain_pick` | `skills/morning_chain.py:310` | neither | — | 1 |
| `morning_handoff` | `skills/morning_handoff.py:736` | neither | yes | 2 |
| `predictive_morning_setup`, `setup_workspace`, `workspace_setup` | `skills/morning_handoff.py:744` | **VERBATIM** | yes | 2 |
| `calendar_next`, `calendar_today`, `ms_graph_calendar` | `skills/ms_graph.py:796` | **VERBATIM** | yes | 4 |
| `list_wifi_clients`, `network_clients`, `who_is_on_the_wifi`, `who_is_on_wifi` | `skills/network_deco.py:693` | *INFORMATIVE* | yes | 2 |
| `is_printer_online`, `printer_online` | `skills/network_deco.py:709` | **VERBATIM** | yes | 2 |
| `device_online`, `is_device_online` | `skills/network_deco.py:729` | **VERBATIM** | yes | 2 |
| `bandwidth_hogs`, `network_usage`, `whats_using_bandwidth` | `skills/network_deco.py:751` | **VERBATIM** | — | 2 |
| `disable_guest_network`, `kick_guest_network` | `skills/network_deco.py:819` | neither | yes | 1 |
| `enable_guest_network` | `skills/network_deco.py:823` | neither | — | 1 |
| `deco_topology` | `skills/network_deco.py:827` | **VERBATIM** | — | 1 |
| `network_topology` | `skills/network_deco.py:827` | *INFORMATIVE* | — | 2 |
| `deco_status` | `skills/network_deco.py:843` | **VERBATIM** | — | 2 |
| `deco_refresh`, `refresh_network` | `skills/network_deco.py:857` | **VERBATIM** | — | 2 |
| `news_briefing` | `skills/news_briefing.py:382` | neither | yes | 4 |
| `enable_night_owl`, `night_owl_mode`, `night_owl_on` | `skills/night_owl_mode.py:472` | neither | yes | 3 |
| `disable_night_owl`, `end_night_owl`, `night_owl_off` | `skills/night_owl_mode.py:475` | neither | yes | 1 |
| `good_morning` | `skills/night_owl_mode.py:478` | **VERBATIM** | — | 2 |
| `night_owl_status` | `skills/night_owl_mode.py:486` | **VERBATIM** | yes | 2 |
| `notification_triage_status`, `triage_status` | `skills/notification_triage.py:1363` | **VERBATIM** | — | 2 |
| `list_notification_rules` | `skills/notification_triage.py:1389` | **VERBATIM** | — | 2 |
| `add_notification_rule` | `skills/notification_triage.py:1398` | **VERBATIM** | — | 2 |
| `remove_notification_rule` | `skills/notification_triage.py:1420` | **VERBATIM** | — | 2 |
| `list_recent_notifications`, `recent_notifications_summary` | `skills/notification_triage.py:1434` | *INFORMATIVE* | yes | 1 |
| `pause_notification_triage` | `skills/notification_triage.py:1452` | **VERBATIM** | — | 2 |
| `resume_notification_triage` | `skills/notification_triage.py:1456` | **VERBATIM** | — | 2 |
| `obs_start_recording` | `skills/obs_control.py:119` | **VERBATIM** | — | 2 |
| `obs_stop_recording` | `skills/obs_control.py:138` | **VERBATIM** | — | 2 |
| `obs_pause_recording` | `skills/obs_control.py:155` | **VERBATIM** | — | 2 |
| `obs_switch_scene` | `skills/obs_control.py:189` | **VERBATIM** | yes | 2 |
| `obs_toggle_mute` | `skills/obs_control.py:230` | **VERBATIM** | yes | 2 |
| `pattern_predictions` | `skills/pattern_learning.py:1069` | **VERBATIM** | yes | 2 |
| `pattern_offer_now` | `skills/pattern_learning.py:1087` | **VERBATIM** | — | 2 |
| `pattern_aggregate` | `skills/pattern_learning.py:1092` | **VERBATIM** | — | 2 |
| `weekly_digest` | `skills/pattern_learning.py:1101` | **VERBATIM** | — | 2 |
| `pattern_stats` | `skills/pattern_learning.py:1115` | **VERBATIM** | — | 2 |
| `rag_search` | `skills/personal_rag.py:118` | **VERBATIM** | yes | 1 |
| `rag_search_quiet` | `skills/personal_rag.py:135` | neither | — | 1 |
| `search_my_files` | `skills/personal_rag.py:135` | **VERBATIM** | — | 2 |
| `rag_reindex` | `skills/personal_rag.py:175` | **VERBATIM** | — | 2 |
| `rag_status` | `skills/personal_rag.py:193` | **VERBATIM** | — | 2 |
| `rag_configure` | `skills/personal_rag.py:212` | **VERBATIM** | — | 2 |
| `rag_open_top` | `skills/personal_rag.py:247` | **VERBATIM** | yes | 2 |
| `notify_phone`, `push_to_phone`, `text_my_phone` | `skills/phone_bridge.py:892` | **VERBATIM** | yes | 2 |
| `phone_bridge_status`, `phone_status` | `skills/phone_bridge.py:925` | **VERBATIM** | — | 2 |
| `list_phone_backends` | `skills/phone_bridge.py:957` | **VERBATIM** | — | 2 |
| `pause_phone_bridge` | `skills/phone_bridge.py:979` | neither | — | 1 |
| `resume_phone_bridge` | `skills/phone_bridge.py:984` | neither | — | 1 |
| `print_companion_status` | `skills/proactive_print_companion.py:736` | **VERBATIM** | — | 2 |
| `print_companion_history` | `skills/proactive_print_companion.py:760` | neither | — | 1 |
| `robot_status` | `skills/repo_robot.py:195` | **VERBATIM** | — | 2 |
| `robot_blocker` | `skills/repo_robot.py:234` | **VERBATIM** | yes | 1 |
| `next_robot_step` | `skills/repo_robot.py:254` | **VERBATIM** | yes | 1 |
| `schedule_cron`, `schedule_recurring` | `skills/schedule_manager.py:147` | **VERBATIM** | yes | 3 |
| `schedule_once` | `skills/schedule_manager.py:238` | **VERBATIM** | yes | 1 |
| `schedule_when`, `when_condition` | `skills/schedule_manager.py:264` | **VERBATIM** | yes | 1 |
| `list_schedule`, `list_schedules`, `show_schedules` | `skills/schedule_manager.py:295` | **VERBATIM** | — | 0 |
| `cancel_schedule`, `remove_schedule` | `skills/schedule_manager.py:310` | **VERBATIM** | — | 1 |
| `fire_schedule`, `run_schedule` | `skills/schedule_manager.py:325` | **VERBATIM** | — | 1 |
| `schedule_status` | `skills/schedule_manager.py:337` | **VERBATIM** | — | 1 |
| `screen_watch_status` | `skills/screen_watch.py:323` | **VERBATIM** | — | 2 |
| `are_you_ok`, `self_diagnostic`, `system_check` | `skills/self_diagnostic.py:2840` | **VERBATIM** | — | 2 |
| `what_is_broken`, `whats_broken` | `skills/self_diagnostic.py:2955` | **VERBATIM** | — | 2 |
| `diagnostic_history` | `skills/self_diagnostic.py:2990` | **VERBATIM** | — | 2 |
| `last_diagnostic_run` | `skills/self_diagnostic.py:3015` | **VERBATIM** | — | 2 |
| `ecobee_request_pin` | `skills/sh_ecobee.py:332` | **VERBATIM** | — | 1 |
| `ecobee_complete_setup` | `skills/sh_ecobee.py:347` | **VERBATIM** | — | 2 |
| `ecobee_authorize` | `skills/sh_ecobee.py:385` | **VERBATIM** | — | 1 |
| `ecobee_list_devices` | `skills/sh_ecobee.py:413` | *INFORMATIVE* | — | 1 |
| `govee_list`, `govee_list_devices` | `skills/sh_govee.py:388` | *INFORMATIVE* | — | 1 |
| `hue_retry_connect` | `skills/sh_hue.py:292` | **VERBATIM** | — | 1 |
| `hue_list`, `hue_list_devices` | `skills/sh_hue.py:463` | *INFORMATIVE* | yes | 1 |
| `hue_set_bridge_ip` | `skills/sh_hue.py:477` | neither | — | 1 |
| `kasa_list`, `kasa_list_devices`, `tplink_list` | `skills/sh_kasa.py:319` | *INFORMATIVE* | — | 1 |
| `lifx_list`, `lifx_list_devices` | `skills/sh_lifx.py:226` | *INFORMATIVE* | — | 1 |
| `nest_authorize` | `skills/sh_nest.py:386` | **VERBATIM** | — | 1 |
| `nest_list_devices` | `skills/sh_nest.py:409` | *INFORMATIVE* | — | 1 |
| `ring_authorize` | `skills/sh_ring.py:332` | **VERBATIM** | — | 1 |
| `ring_list_devices` | `skills/sh_ring.py:406` | *INFORMATIVE* | — | 1 |
| `smart_life_list`, `tuya_list`, `tuya_list_devices` | `skills/sh_tuya.py:157` | *INFORMATIVE* | — | 1 |
| `discover_smart_home`, `refresh_smart_home`, `smart_home_discover`, `smart_home_setup` | `skills/smart_home_discover.py:1141` | neither | — | 1 |
| `list_smart_home_devices`, `smart_home_catalog` | `skills/smart_home_discover.py:1418` | **VERBATIM** | — | 2 |
| `forget_alexa_login`, `smart_home_purge_cookie` | `skills/smart_home_discover.py:1434` | neither | — | 1 |
| `last_gate_result`, `last_stability_gate`, `last_stability_gate_result` | `skills/stability_gate_status.py:56` | **VERBATIM** | — | 2 |
| `audio_music_status` | `skills/standby_audio_detect.py:576` | **VERBATIM** | — | 2 |
| `status_panel`, `suit_diagnostics`, `system_status` | `skills/status_panel.py:511` | **VERBATIM** | yes | 2 |
| `suit_up`, `suit_up_sequence` | `skills/suit_up.py:361` | neither | yes | 1 |
| `check_system` | `skills/system_monitor.py:249` | **VERBATIM** | yes | 3 |
| `status_report`, `system_pulse` | `skills/system_pulse.py:666` | **VERBATIM** | yes | 5 |
| `check_teams` | `skills/teams_nudge.py:203` | **VERBATIM** | yes | 3 |
| `screen_teams_calls` | `skills/teams_screener.py:571` | **VERBATIM** | — | 0 |
| `answer_call` | `skills/teams_screener.py:582` | **VERBATIM** | yes | 0 |
| `decline_call` | `skills/teams_screener.py:597` | **VERBATIM** | yes | 0 |
| `vip_priority_handler` | `skills/teams_screener.py:687` | neither | yes | 0 |
| `set_timer` | `skills/timer.py:364` | neither | yes | 9 |
| `list_timers` | `skills/timer.py:401` | **VERBATIM** | — | 1 |
| `cancel_timer` | `skills/timer.py:427` | **VERBATIM** | yes | 2 |
| `bonnaroo_brief` | `skills/trip_planner.py:622` | neither | — | 0 |
| `bonnaroo_status` | `skills/trip_planner.py:622` | **VERBATIM** | — | 1 |
| `bonnaroo_prep` | `skills/trip_planner.py:630` | neither | — | 0 |
| `calibrate_tv_region`, `tv_calibrate` | `skills/tv_detect.py:303` | **VERBATIM** | — | 1 |
| `tv_detect_status`, `tv_status` | `skills/tv_detect.py:348` | **VERBATIM** | — | 2 |
| `tv_detect_on` | `skills/tv_detect.py:381` | **VERBATIM** | yes | 1 |
| `tv_detect_off` | `skills/tv_detect.py:399` | **VERBATIM** | — | 1 |
| `holographic_status` | `skills/vip_boss_mode.py:350` | **VERBATIM** | — | 2 |
| `vip_intercept_status` | `skills/vip_boss_mode.py:350` | **VERBATIM** | — | 1 |
| `wayne_boss_mode_status` | `skills/vip_boss_mode.py:350` | **VERBATIM** | — | 1 |
| `wayne_boss_test_work_hours` | `skills/vip_boss_mode.py:364` | neither | — | 0 |
| `wayne_boss_test_evening` | `skills/vip_boss_mode.py:375` | neither | — | 0 |
| `send_vip_reply` | `skills/vip_intercept.py:654` | **VERBATIM** | — | 1 |
| `scrap_vip_reply` | `skills/vip_intercept.py:674` | **VERBATIM** | — | 0 |
| `vip_intercept_test` | `skills/vip_intercept.py:693` | neither | — | 0 |
| `list_voice_profiles` | `skills/voice_clone.py:123` | **VERBATIM** | yes | 1 |
| `set_voice_profile`, `switch_voice_profile`, `use_voice_profile` | `skills/voice_clone.py:148` | **VERBATIM** | yes | 1 |
| `voice_clone_status` | `skills/voice_clone.py:182` | **VERBATIM** | yes | 1 |
| `disable_voice_clone`, `stop_voice_clone`, `voice_clone_off` | `skills/voice_clone.py:203` | **VERBATIM** | yes | 1 |
| `wake_listener_start` | `skills/wake_listener.py:403` | neither | — | 1 |
| `wake_listener_stop` | `skills/wake_listener.py:420` | neither | — | 1 |
| `wake_listener_status` | `skills/wake_listener.py:434` | **VERBATIM** | — | 2 |
| `wake_listener_configure` | `skills/wake_listener.py:460` | neither | — | 1 |
| `guest_mode_on` | `skills/wake_listener.py:528` | neither | — | 1 |
| `guest_mode_off` | `skills/wake_listener.py:536` | neither | — | 1 |
| `voice_gating_on` | `skills/wake_listener.py:543` | neither | — | 1 |
| `voice_gating_off` | `skills/wake_listener.py:550` | neither | — | 1 |
| `weather_briefing`, `weather_forecast` | `skills/weather_briefing.py:504` | **VERBATIM** | yes | 6 |
| `web_interface_on` | `skills/web_interface.py:205` | **VERBATIM** | yes | 1 |
| `web_interface_off` | `skills/web_interface.py:209` | **VERBATIM** | yes | 1 |
| `web_interface_status` | `skills/web_interface.py:213` | **VERBATIM** | yes | 1 |
| `weekly_digest_now` | `skills/weekly_digest_briefing.py:483` | neither | — | 1 |
| `weekly_digest_status` | `skills/weekly_digest_briefing.py:501` | **VERBATIM** | — | 2 |
| `wellness_status` | `skills/wellness.py:331` | **VERBATIM** | yes | 1 |
| `workshop_status` | `skills/workshop_mode.py:287` | **VERBATIM** | — | 2 |
| `youtube_direct`, `youtube_search_direct`, `yt_direct` | `skills/youtube_search.py:195` | **VERBATIM** | — | 2 |
