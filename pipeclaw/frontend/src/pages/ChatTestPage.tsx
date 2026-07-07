import { ChatPanel } from '../components/ChatPanel';

export default function ChatTestPage() {
  return (
    <div style={{ width: '100%', maxWidth: '900px', height: '100vh', margin: '0 auto' }}>
      <ChatPanel currentDate="2019-01-01" selectedItem={{ type: 'system', id: '西气东输二线' }} />
    </div>
  );
}
