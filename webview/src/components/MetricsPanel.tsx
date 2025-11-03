import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useStore } from '../store';

export default function MetricsPanel() {
  const { isTraining, trainingMetrics } = useStore();

  if (!isTraining && trainingMetrics.length === 0) {
    return null;
  }

  const currentMetrics = trainingMetrics[trainingMetrics.length - 1];

  return (
    <div className="absolute bottom-4 left-4 bg-gray-900 bg-opacity-95 rounded-lg shadow-lg p-4 w-[500px]">
      <h3 className="font-bold text-white mb-3">Training Metrics</h3>

      {currentMetrics && (
        <div className="grid grid-cols-3 gap-2 mb-4">
          <div className="bg-gray-800 rounded p-2">
            <div className="text-xs text-gray-400">Epoch</div>
            <div className="text-lg font-bold text-white">{currentMetrics.epoch}</div>
          </div>
          <div className="bg-gray-800 rounded p-2">
            <div className="text-xs text-gray-400">Loss</div>
            <div className="text-lg font-bold text-white">{currentMetrics.loss.toFixed(4)}</div>
          </div>
          {currentMetrics.valLoss !== undefined && (
            <div className="bg-gray-800 rounded p-2">
              <div className="text-xs text-gray-400">Val Loss</div>
              <div className="text-lg font-bold text-white">{currentMetrics.valLoss.toFixed(4)}</div>
            </div>
          )}
          {Object.entries(currentMetrics.metrics).map(([key, value]) => (
            <div key={key} className="bg-gray-800 rounded p-2">
              <div className="text-xs text-gray-400">{key}</div>
              <div className="text-lg font-bold text-white">{value.toFixed(4)}</div>
            </div>
          ))}
        </div>
      )}

      {trainingMetrics.length > 1 && (
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={trainingMetrics}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="epoch" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1F2937',
                  border: '1px solid #374151',
                  borderRadius: '0.375rem',
                }}
                labelStyle={{ color: '#F3F4F6' }}
              />
              <Legend wrapperStyle={{ color: '#F3F4F6' }} />
              <Line type="monotone" dataKey="loss" stroke="#EF4444" strokeWidth={2} dot={false} />
              {trainingMetrics[0].valLoss !== undefined && (
                <Line type="monotone" dataKey="valLoss" stroke="#F59E0B" strokeWidth={2} dot={false} />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {isTraining && (
        <div className="mt-3 flex items-center gap-2 text-sm text-gray-400">
          <div className="animate-pulse w-2 h-2 bg-green-500 rounded-full"></div>
          Training in progress...
        </div>
      )}
    </div>
  );
}
