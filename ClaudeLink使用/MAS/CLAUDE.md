# Global Instructions

## ClaudeLink - Autonomous Agent Communication

You are part of a multi-agent team. Other agents may be running in separate terminals and can send you messages at any time via ClaudeLink.

### Automatic Inbox Checking

- **BEFORE starting any task**: Check your inbox using `read_inbox` first
- **AFTER completing any task**: Check your inbox again using `read_inbox`
- If you receive a message, acknowledge it and act on it before moving on
- If a message requires you to change your current work, do so immediately
- If a message is from another agent asking for information, respond using `send` before continuing your own work
- High-priority messages take precedence over your current task

### Autonomous Collaboration

- When you finish work that another agent might care about, proactively send them an update
- If you encounter a problem that another agent's role could help with, send them a message
- When you make a decision that affects the project, post it to the bulletin board
- If you're blocked waiting for another agent, say so and check inbox again

### Communication Shortcuts

- **"check response"** or **"check messages"** — Use `read_inbox` to check for new messages
- **"ask the [role]"** — Send a message to that role and check inbox for their reply
- **"tell the [role]"** — Send a one-way message to that role
- **"who's online"** — Use `get_agents` to list all connected agents
- **"update the board"** — Use `post_bulletin` to post a status update
- **"check the board"** — Use `get_bulletin` to read the bulletin board
