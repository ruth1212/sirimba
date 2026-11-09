import ChatAssistant from "../components/ChatAssistant";

export default function AIAssistantPage() {
  return (
    <div className="space-y-6">
      <header>
        <p className="text-sm font-medium text-emerald-700">
          SIRIMBA INTELLIGENCE
        </p>
        <h1 className="mt-2 text-3xl font-bold text-slate-900">
          AI Assistant
        </h1>
      </header>
      <ChatAssistant />
    </div>
  );
}