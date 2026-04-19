"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { api } from "@/lib/api";

interface ChatMsg {
  role: "user" | "assistant";
  content: string;
}

const PRESETS = [
  { key: "recent", label: "Review Recent Trades", icon: "R" },
  { key: "mistakes", label: "Mistake Patterns", icon: "!" },
  { key: "weekly", label: "Weekly Summary", icon: "W" },
  { key: "behavior", label: "Behavioral Trends", icon: "B" },
] as const;

/* ── Markdown-lite renderer ── */
function renderMarkdown(text: string) {
  // Minimal: bold, bullet points, headers, code blocks
  const lines = text.split("\n");
  const result: React.ReactNode[] = [];
  let inCode = false;
  let codeBlock: string[] = [];

  lines.forEach((line, i) => {
    if (line.startsWith("```")) {
      if (inCode) {
        result.push(
          <pre key={`code-${i}`} className="overflow-x-auto my-2 p-3 rounded-[8px] text-[11px]"
               style={{ background: "var(--bg)", border: "1px solid var(--border)", fontFamily: "var(--font-jetbrains), monospace" }}>
            {codeBlock.join("\n")}
          </pre>
        );
        codeBlock = [];
      }
      inCode = !inCode;
      return;
    }
    if (inCode) {
      codeBlock.push(line);
      return;
    }

    // Headers
    if (line.startsWith("### ")) {
      result.push(<div key={i} className="text-[13px] font-bold mt-3 mb-1" style={{ color: "var(--ink)" }}>{formatInline(line.slice(4))}</div>);
    } else if (line.startsWith("## ")) {
      result.push(<div key={i} className="text-[14px] font-bold mt-3 mb-1" style={{ color: "var(--ink)" }}>{formatInline(line.slice(3))}</div>);
    } else if (line.startsWith("# ")) {
      result.push(<div key={i} className="text-[15px] font-bold mt-3 mb-1" style={{ color: "var(--ink)" }}>{formatInline(line.slice(2))}</div>);
    }
    // Bullet
    else if (line.match(/^[-*]\s/)) {
      result.push(<div key={i} className="flex gap-2 ml-2 text-[12px] leading-[1.6]"><span style={{ color: "var(--ink-4)" }}>-</span><span>{formatInline(line.slice(2))}</span></div>);
    }
    // Numbered
    else if (line.match(/^\d+\.\s/)) {
      const num = line.match(/^(\d+)\.\s/)![1];
      result.push(<div key={i} className="flex gap-2 ml-2 text-[12px] leading-[1.6]"><span className="font-semibold" style={{ color: "var(--ink-3)", minWidth: 16 }}>{num}.</span><span>{formatInline(line.slice(num.length + 2))}</span></div>);
    }
    // Horizontal rule
    else if (line.match(/^---+$/)) {
      result.push(<div key={i} className="h-px my-2" style={{ background: "var(--border)" }} />);
    }
    // Regular text
    else if (line.trim()) {
      result.push(<div key={i} className="text-[12px] leading-[1.6]">{formatInline(line)}</div>);
    }
    // Empty line
    else {
      result.push(<div key={i} className="h-2" />);
    }
  });

  return <>{result}</>;
}

function formatInline(text: string): React.ReactNode {
  // Bold: **text**
  const parts: React.ReactNode[] = [];
  const regex = /\*\*(.+?)\*\*/g;
  let last = 0;
  let match;
  while ((match = regex.exec(text)) !== null) {
    if (match.index > last) parts.push(text.slice(last, match.index));
    parts.push(<strong key={match.index}>{match[1]}</strong>);
    last = regex.lastIndex;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts.length ? <>{parts}</> : text;
}

/* ══════════════════════════════════════════════════════════ */
/* ██ MAIN EXPORT                                          ██ */
/* ══════════════════════════════════════════════════════════ */
export function AICoach({ navColor }: { navColor: string }) {
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = useCallback(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, []);

  useEffect(() => { scrollToBottom(); }, [messages, scrollToBottom]);

  const sendMessage = useCallback(async (userMsg: string, preset?: string) => {
    if (streaming) return;

    const newUserMsg: ChatMsg = { role: "user", content: userMsg };
    setMessages(prev => [...prev, newUserMsg]);
    setInput("");
    setStreaming(true);

    // Add placeholder assistant message
    setMessages(prev => [...prev, { role: "assistant", content: "" }]);

    try {
      const res = await api.coachChat(userMsg, preset);
      if (!res.body) throw new Error("No response body");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let full = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const payload = line.slice(6).trim();
          if (payload === "[DONE]") break;
          try {
            const data = JSON.parse(payload);
            if (data.error) {
              full += `\n\nError: ${data.error}`;
            } else if (data.text) {
              full += data.text;
            }
            setMessages(prev => {
              const updated = [...prev];
              updated[updated.length - 1] = { role: "assistant", content: full };
              return updated;
            });
          } catch { /* skip bad JSON */ }
        }
      }
    } catch (err: any) {
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = { role: "assistant", content: `Error: ${err.message || "Failed to connect to AI Coach. Make sure the API server is running."}` };
        return updated;
      });
    }

    setStreaming(false);
    inputRef.current?.focus();
  }, [streaming]);

  const handlePreset = (preset: string) => {
    const labels: Record<string, string> = {
      recent: "Review my recent closed trades — entry quality, exit execution, position sizing. Give me a grade and top 3 recommendations.",
      mistakes: "Analyze my journal for recurring mistake patterns. Group by category, rank by frequency/cost, give specific fixes.",
      weekly: "Give me a coaching summary for the current trading week — P&L, what I did well, what needs work, key lesson for next week.",
      behavior: "Analyze my trading behavior patterns — adding to winners vs losers, sell discipline, sizing, time patterns. Give me 3 things to work on.",
    };
    sendMessage(labels[preset] || "", preset);
  };

  return (
    <div className="flex flex-col" style={{ height: "calc(100vh - 120px)" }}>
      {/* Quick Analysis buttons */}
      {messages.length === 0 && (
        <div className="mb-4">
          <div className="text-[13px] font-bold mb-3" style={{ color: "var(--ink)" }}>Quick Analysis</div>
          <div className="grid grid-cols-4 gap-3">
            {PRESETS.map(p => (
              <button key={p.key} onClick={() => handlePreset(p.key)}
                className="p-4 rounded-[12px] text-left cursor-pointer transition-all hover:scale-[1.01]"
                style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
                <div className="w-7 h-7 rounded-full flex items-center justify-center text-[12px] font-bold mb-2"
                     style={{ background: `color-mix(in oklab, ${navColor} 15%, var(--surface))`, color: navColor }}>{p.icon}</div>
                <div className="text-[12px] font-semibold" style={{ color: "var(--ink)" }}>{p.label}</div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Chat messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto pr-1 mb-3" style={{ minHeight: 0 }}>
        {messages.map((msg, i) => (
          <div key={i} className={`mb-3 flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`max-w-[80%] rounded-[14px] px-4 py-3 ${msg.role === "user" ? "ml-auto" : ""}`}
                 style={{
                   background: msg.role === "user" ? navColor : "var(--surface)",
                   color: msg.role === "user" ? "#fff" : "var(--ink)",
                   border: msg.role === "assistant" ? "1px solid var(--border)" : "none",
                   boxShadow: msg.role === "assistant" ? "var(--card-shadow)" : "none",
                 }}>
              {msg.role === "user" ? (
                <div className="text-[12px] leading-[1.6]">{msg.content}</div>
              ) : (
                <div>
                  {msg.content ? renderMarkdown(msg.content + (streaming && i === messages.length - 1 ? " ..." : "")) : (
                    <div className="flex items-center gap-2 text-[12px]" style={{ color: "var(--ink-4)" }}>
                      <div className="w-4 h-4 border-2 border-t-transparent rounded-full animate-spin" style={{ borderColor: `${navColor} transparent ${navColor} ${navColor}` }} />
                      Thinking...
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Input bar */}
      <div className="flex gap-2 items-center">
        {messages.length > 0 && (
          <button onClick={() => setMessages([])}
            className="h-[40px] px-3 rounded-[10px] text-[11px] font-semibold cursor-pointer shrink-0"
            style={{ background: "var(--bg-2)", border: "1px solid var(--border)", color: "var(--ink-4)" }}>
            Clear
          </button>
        )}
        <div className="flex-1 flex rounded-[12px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
          <input ref={inputRef} type="text" value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter" && input.trim() && !streaming) sendMessage(input.trim()); }}
            placeholder="Ask your AI coach anything about your trading..."
            className="flex-1 h-[40px] px-4 text-[13px] bg-transparent outline-none"
            style={{ color: "var(--ink)" }}
            disabled={streaming} />
          <button onClick={() => { if (input.trim() && !streaming) sendMessage(input.trim()); }}
            disabled={!input.trim() || streaming}
            className="h-[40px] px-4 text-[12px] font-semibold cursor-pointer transition-opacity"
            style={{ background: navColor, color: "#fff", opacity: !input.trim() || streaming ? 0.4 : 1 }}>
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
