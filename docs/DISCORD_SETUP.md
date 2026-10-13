# Discord setup

JARVIS can reach you on Discord: alerts arrive as direct messages from a bot,
and you can DM the bot to talk to JARVIS (each DM is handled like a typed
command and answered in the same DM). Discord is the primary phone channel;
ntfy stays as the backup.

Only your own Discord account can command the bot. Anyone else who messages
it, or any message in a server channel, is ignored without a reply.

## 1. Create the bot

1. Open the Discord developer portal (discord.com/developers/applications) and
   click **New Application**. Name it anything.
2. Open the **Bot** tab. Click **Reset Token** and copy the token. Treat it
   like a password: it is only ever stored in an environment variable, never
   in the repo, the settings file or a log.
3. On the same tab, under **Privileged Gateway Intents**, turn on
   **Message Content Intent** and save.
4. On the **Bot** tab you may turn **Public Bot** off so nobody else can add it.

## 2. Invite it and let it DM you

A bot can only DM someone it shares a server with.

1. Open **OAuth2 > URL Generator**. Tick the **bot** scope. No permissions are
   needed for DMs (leave them all unticked).
2. Open the generated URL, choose a server you own (a private one with only
   you in it is ideal) and authorise.
3. In Discord, make sure **Settings > Content & Social > Server members can
   send you direct messages** is on for that server (or right-click the
   server > Privacy Settings > Direct Messages).

## 3. Find your user id

1. In Discord open **Settings > Advanced** and turn on **Developer Mode**.
2. Right-click your own name and choose **Copy User ID**. It is a long number.

## 4. Set the two environment variables

JARVIS reads them from the environment of the process that starts it.

PowerShell (user level, persists across reboots):

```powershell
[System.Environment]::SetEnvironmentVariable('DISCORD_BOT_TOKEN', '<token>', 'User')
[System.Environment]::SetEnvironmentVariable('DISCORD_OWNER_ID',  '<your user id>', 'User')
# optional backup channel, already supported:
[System.Environment]::SetEnvironmentVariable('NTFY_TOPIC', '<long random topic>', 'User')
```

If JARVIS is started by a launcher script that forwards user-level variables
to the child process by name (the way the Anthropic key and printer
credentials are forwarded), add `DISCORD_BOT_TOKEN`, `DISCORD_OWNER_ID` and
`NTFY_TOPIC` to that script's list, for example:

```powershell
foreach ($v in 'DISCORD_BOT_TOKEN','DISCORD_OWNER_ID','NTFY_TOPIC') {
    $val = [System.Environment]::GetEnvironmentVariable($v, 'User')
    if (-not [string]::IsNullOrWhiteSpace($val)) { Set-Item -Path "env:$v" -Value $val }
}
```

Restart JARVIS. The boot log shows `[discord] gateway connected` once the bot is
online, and the bot shows as online in your server's member list.

## 5. Try it

* Say "send a test ping" at the desk. The test arrives as a DM from the bot.
* DM the bot "what time is it" or any command you would type to JARVIS. The
  reply comes back in the DM. `/help` and `/status` work too.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `DISCORD_ENABLED` | on | Master switch for sending and listening. |
| `DISCORD_INBOUND` | on | Listen for your DMs. Off = alerts out only. |
| `DISCORD_NTFY_FALLBACK_ONLY` | on | Alerts go to Discord and ntfy is used only if the Discord send fails. Off = both. |

Nothing happens until both environment variables are set.

## Troubleshooting

* **Nothing arrives, log says the token was rejected**: the token was reset or
  mistyped. Reset it in the portal and update the variable.
* **Log says a privileged intent is not enabled**: turn on **Message Content
  Intent** (step 1.3).
* **Alerts arrive but DMs get no answer**: check the boot log for
  `DM listener not started`. A staging or test instance never listens, so two
  instances cannot answer the same DM.
* **Sending fails with HTTP 403**: the bot shares no server with you, or your
  server privacy setting blocks DMs (step 2).
